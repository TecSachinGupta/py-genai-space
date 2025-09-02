import numpy as np
import json
import os
import logging
from typing import List, Dict, Any, Optional, Set
import uuid

import chromadb
from src.schemas.vector_database import VectorDatabase

logger = logging.getLogger(__name__)

class ChromaVectorDB(VectorDatabase):
    """Enhanced ChromaDB-based vector database with duplicate prevention"""
    
    def __init__(self, collection_name: str = "documents", 
                 persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.document_hashes: Set[str] = set()
        
        # Create persist directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Metadata file for document hashes
        self.metadata_file = os.path.join(persist_directory, "document_metadata.json")
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=collection_name)
            except:
                self.collection = self.client.create_collection(name=collection_name)
            
            # Load existing document hashes
            self._load_document_hashes()
            
            logger.info(f"Enhanced ChromaDB initialized: {collection_name} at {persist_directory}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise e
    
    def _load_document_hashes(self):
        """Load document hashes from metadata file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.document_hashes = set(data.get('document_hashes', []))
                logger.info(f"Loaded {len(self.document_hashes)} document hashes from metadata")
            except Exception as e:
                logger.warning(f"Error loading document metadata: {e}")
                self.document_hashes = set()
        
        # Also try to load from existing collection metadata
        try:
            # Get all documents to rebuild hash set if needed
            if len(self.document_hashes) == 0:
                result = self.collection.get(include=['metadatas'])
                if result['metadatas']:
                    for metadata in result['metadatas']:
                        if 'document_hash' in metadata:
                            self.document_hashes.add(metadata['document_hash'])
                    logger.info(f"Rebuilt {len(self.document_hashes)} document hashes from collection")
        except Exception as e:
            logger.warning(f"Error rebuilding hashes from collection: {e}")
    
    def _save_document_hashes(self):
        """Save document hashes to metadata file."""
        try:
            metadata = {
                'document_hashes': list(self.document_hashes),
                'total_documents': len(self.document_hashes),
                'collection_name': self.collection_name
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved {len(self.document_hashes)} document hashes to metadata")
        except Exception as e:
            logger.error(f"Error saving document metadata: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]], check_duplicates: bool = True) -> List[str]:
        """Add documents to ChromaDB with optional duplicate checking"""
        if check_duplicates:
            unique_docs, duplicate_count = self._filter_duplicate_documents(documents, self.document_hashes)
            if duplicate_count > 0:
                logger.info(f"Filtered out {duplicate_count} duplicate documents")
            if not unique_docs:
                logger.info("No new unique documents to add")
                return []
            documents = unique_docs
        
        doc_ids = []
        embeddings = []
        texts = []
        metadatas = []
        
        try:
            for doc in documents:
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                # Extract embedding
                embedding = doc['embedding']
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                embeddings.append(embedding)
                
                # Extract text
                texts.append(doc.get('text', ''))
                
                # Prepare metadata
                metadata = doc.copy()
                del metadata['embedding']
                
                # Add document hash to metadata
                if check_duplicates:
                    doc_hash = self._generate_document_hash(doc)
                    metadata['document_hash'] = doc_hash
                    self.document_hashes.add(doc_hash)
                
                # ChromaDB metadata values must be strings, numbers, or booleans
                clean_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)
                
                metadatas.append(clean_metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            if check_duplicates:
                self._save_document_hashes()
            
            logger.info(f"Added {len(doc_ids)} documents to ChromaDB")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise e
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using ChromaDB"""
        try:
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i] if results['documents'] else '',
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1.0 - results['distances'][0][i] if results['distances'] else 0.0  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during ChromaDB search: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from ChromaDB"""
        try:
            # Get document metadata first to remove hash
            doc = self.get_document(doc_id)
            
            self.collection.delete(ids=[doc_id])
            
            # Remove from hash set if it has a hash
            if doc and 'metadata' in doc and 'document_hash' in doc['metadata']:
                doc_hash = doc['metadata']['document_hash']
                if doc_hash in self.document_hashes:
                    self.document_hashes.remove(doc_hash)
                    self._save_document_hashes()
            
            logger.info(f"Document {doc_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            result = self.collection.get(
                ids=[doc_id], 
                include=['documents', 'metadatas']
            )
            
            if result['ids'] and result['ids'][0]:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0] if result['documents'] else '',
                    'metadata': result['metadatas'][0] if result['metadatas'] else {}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update document in ChromaDB"""
        try:
            # ChromaDB doesn't have direct update, so we delete and add
            old_doc = self.get_document(doc_id)
            if not old_doc:
                return False
            
            self.collection.delete(ids=[doc_id])
            
            # Prepare document data
            embedding = document['embedding']
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            text = document.get('text', '')
            metadata = document.copy()
            del metadata['embedding']
            
            # Keep the same document hash if it exists
            if old_doc and 'metadata' in old_doc and 'document_hash' in old_doc['metadata']:
                metadata['document_hash'] = old_doc['metadata']['document_hash']
            
            # Clean metadata
            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)
            
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[clean_metadata]
            )
            
            logger.info(f"Document {doc_id} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'tracked_hashes': len(self.document_hashes)
            }
        except Exception as e:
            return {
                'error': str(e),
                'collection_name': self.collection_name
            }
    
    def save_store(self, filepath: Optional[str] = None) -> bool:
        """Save ChromaDB metadata (ChromaDB auto-persists)"""
        try:
            self._save_document_hashes()
            logger.info("ChromaDB metadata saved (collection auto-persisted)")
            return True
        except Exception as e:
            logger.error(f"Error saving ChromaDB metadata: {e}")
            return False
    
    def load_store(self, filepath: str) -> bool:
        """Load ChromaDB from directory"""
        try:
            self.persist_directory = filepath
            self.metadata_file = os.path.join(filepath, "document_metadata.json")
            
            # Reinitialize client
            self.client = chromadb.PersistentClient(path=filepath)
            self.collection = self.client.get_collection(name=self.collection_name)
            
            # Reload document hashes
            self._load_document_hashes()
            
            logger.info(f"ChromaDB loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ChromaDB from {filepath}: {e}")
            return False