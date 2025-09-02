import numpy as np
import logging
from typing import List, Dict, Any, Optional, Set
import uuid

from src.schemas.vector_database import VectorDatabase

logger = logging.getLogger(__name__)

class InMemoryVectorDB(VectorDatabase):
    """in-memory vector database with duplicate prevention"""
    
    def __init__(self):
        self.documents = {}
        self.embeddings = {}
        self.document_hashes: Set[str] = set()
        logger.info("Enhanced InMemory Vector DB initialized")
    
    def add_documents(self, documents: List[Dict[str, Any]], check_duplicates: bool = True) -> List[str]:
        """Add documents to memory with optional duplicate checking"""
        if check_duplicates:
            unique_docs, duplicate_count = self._filter_duplicate_documents(documents, self.document_hashes)
            if duplicate_count > 0:
                logger.info(f"Filtered out {duplicate_count} duplicate documents")
            if not unique_docs:
                logger.info("No new unique documents to add")
                return []
            documents = unique_docs
        
        doc_ids = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            # Store document
            doc_copy = doc.copy()
            embedding = doc_copy.pop('embedding')
            
            # Add document hash
            if check_duplicates:
                doc_hash = self._generate_document_hash(doc)
                doc_copy['document_hash'] = doc_hash
                self.document_hashes.add(doc_hash)
            
            self.documents[doc_id] = doc_copy
            self.embeddings[doc_id] = np.array(embedding)
        
        logger.info(f"Added {len(doc_ids)} documents to in-memory database")
        return doc_ids
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using cosine similarity"""
        if not self.embeddings:
            return []
        
        similarities = []
        for doc_id, embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
            )
            similarities.append((doc_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for doc_id, score in similarities[:top_k]:
            doc = self.documents[doc_id].copy()
            doc['id'] = doc_id
            doc['score'] = float(score)
            results.append(doc)
        
        return results
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from memory"""
        if doc_id in self.documents:
            # Remove from hash set
            doc = self.documents[doc_id]
            if 'document_hash' in doc and doc['document_hash'] in self.document_hashes:
                self.document_hashes.remove(doc['document_hash'])
            
            del self.documents[doc_id]
            del self.embeddings[doc_id]
            logger.info(f"Document {doc_id} deleted from in-memory database")
            return True
        return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if doc_id in self.documents:
            doc = self.documents[doc_id].copy()
            doc['id'] = doc_id
            return doc
        return None
    
    def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update document in memory"""
        if doc_id in self.documents:
            doc_copy = document.copy()
            embedding = doc_copy.pop('embedding', None)
            
            # Keep the same hash if it exists
            old_doc = self.documents[doc_id]
            if 'document_hash' in old_doc:
                doc_copy['document_hash'] = old_doc['document_hash']
            
            self.documents[doc_id] = doc_copy
            if embedding is not None:
                self.embeddings[doc_id] = np.array(embedding)
            
            logger.info(f"Document {doc_id} updated in in-memory database")
            return True
        return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'total_documents': len(self.documents),
            'database_type': 'InMemory',
            'tracked_hashes': len(self.document_hashes)
        }
    
    def save_store(self, filepath: Optional[str] = None) -> bool:
        """Save in-memory database to pickle file"""
        try:
            save_path = filepath or "./inmemory_vectordb.pkl"
            
            data = {
                'documents': self.documents,
                'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
                'document_hashes': list(self.document_hashes)
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"In-memory database saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving in-memory database: {e}")
            return False
    
    def load_store(self, filepath: str) -> bool:
        """Load in-memory database from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data.get('documents', {})
            embeddings_data = data.get('embeddings', {})
            self.embeddings = {k: np.array(v) for k, v in embeddings_data.items()}
            self.document_hashes = set(data.get('document_hashes', []))
            
            logger.info(f"In-memory database loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading in-memory database: {e}")
            return False