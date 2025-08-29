import os
import logging
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Set

# LangChain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaissStore:
    """Enhanced FAISS vector store with improved functionality and duplicate prevention"""
    
    def __init__(self, store_path: str = "./vector_store", embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.store_path = store_path
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.retriever = None
        self.document_hashes: Set[str] = set()
        self.metadata_file = os.path.join(store_path, "document_metadata.json")
        
        # Initialize embeddings
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        logger.info("Embedding model initialized")

    def _generate_document_hash(self, document: Document) -> str:
        """Generate a unique hash for a document based on its content and metadata."""
        # Create a string representation of the document
        content = document.page_content
        # Sort metadata keys to ensure consistent hashing
        metadata_str = json.dumps(document.metadata, sort_keys=True, default=str)
        combined = f"{content}||{metadata_str}"
        
        # Generate SHA-256 hash
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

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
        else:
            self.document_hashes = set()

    def _save_document_hashes(self):
        """Save document hashes to metadata file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            
            metadata = {
                'document_hashes': list(self.document_hashes),
                'total_documents': len(self.document_hashes)
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved {len(self.document_hashes)} document hashes to metadata")
        except Exception as e:
            logger.error(f"Error saving document metadata: {e}")

    def _filter_duplicate_documents(self, documents: List[Document]) -> Tuple[List[Document], int]:
        """Filter out duplicate documents and return unique ones along with duplicate count."""
        unique_documents = []
        duplicate_count = 0
        
        for doc in documents:
            doc_hash = self._generate_document_hash(doc)
            
            if doc_hash not in self.document_hashes:
                unique_documents.append(doc)
                self.document_hashes.add(doc_hash)
            else:
                duplicate_count += 1
                logger.debug(f"Skipping duplicate document with hash: {doc_hash[:16]}...")
        
        return unique_documents, duplicate_count

    def create_load_store(self, documents: Optional[List[Document]] = None) -> FAISS:
        """Create new store or load existing one. If both exist, merge documents with existing store."""
        # Load existing document hashes
        self._load_document_hashes()
        
        if os.path.exists(self.store_path):
            logger.info("Found existing vector store, loading...")
            vector_store = self.load_store()
            
            if documents:
                logger.info(f"Processing {len(documents)} documents for addition to existing store...")
                unique_docs, duplicate_count = self._filter_duplicate_documents(documents)
                
                if duplicate_count > 0:
                    logger.info(f"Filtered out {duplicate_count} duplicate documents")
                
                if unique_docs:
                    logger.info(f"Adding {len(unique_docs)} new unique documents to existing store...")
                    self.add_documents_without_duplicate_check(unique_docs)
                    self.save_store()
                    self._save_document_hashes()
                    logger.info("Unique documents merged and store updated")
                else:
                    logger.info("No new unique documents to add")
            
            return vector_store
        elif documents:
            logger.info("Creating new vector store...")
            unique_docs, duplicate_count = self._filter_duplicate_documents(documents)
            
            if duplicate_count > 0:
                logger.info(f"Filtered out {duplicate_count} duplicate documents during store creation")
            
            if unique_docs:
                vector_store = self.create_store(unique_docs)
                self.save_store()
                self._save_document_hashes()
                return vector_store
            else:
                raise ValueError("No unique documents provided to create new store")
        else:
            raise ValueError("No existing store found and no documents provided to create new store")

    def create_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents."""
        logger.info(f"Creating FAISS vector store from {len(documents)} documents...")
        
        try:
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            logger.info(f"Vector store created successfully")
            logger.info(f"Index size: {self.vector_store.index.ntotal} vectors")
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise e

    def load_store(self):
        """Load FAISS vector store from disk."""
        try:
            self.vector_store = FAISS.load_local(
                self.store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            logger.info(f"Vector store loaded from: {self.store_path}")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None

    def save_store(self):
        """Save FAISS vector store to disk."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        self.vector_store.save_local(self.store_path)
        logger.info(f"Vector store saved to: {self.store_path}")

    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store with duplicate checking."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load a store first.")
        
        logger.info(f"Processing {len(documents)} documents for addition...")
        unique_docs, duplicate_count = self._filter_duplicate_documents(documents)
        
        if duplicate_count > 0:
            logger.info(f"Filtered out {duplicate_count} duplicate documents")
        
        if unique_docs:
            self.add_documents_without_duplicate_check(unique_docs)
            self._save_document_hashes()
        else:
            logger.info("No new unique documents to add")

    def add_documents_without_duplicate_check(self, documents: List[Document]):
        """Add documents to vector store without duplicate checking (internal use)."""
        logger.info(f"Adding {len(documents)} documents to vector store...")
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Documents added successfully")
            logger.info(f"Updated index size: {self.vector_store.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise e

    def force_add_documents(self, documents: List[Document]):
        """Force add documents without duplicate checking (for special cases)."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load a store first.")
        
        logger.warning(f"Force adding {len(documents)} documents without duplicate checking...")
        
        # Update hashes for tracking
        for doc in documents:
            doc_hash = self._generate_document_hash(doc)
            self.document_hashes.add(doc_hash)
        
        self.add_documents_without_duplicate_check(documents)
        self._save_document_hashes()

    def check_document_exists(self, document: Document) -> bool:
        """Check if a document already exists in the store."""
        doc_hash = self._generate_document_hash(document)
        return doc_hash in self.document_hashes

    def get_duplicate_info(self, documents: List[Document]) -> Dict[str, Any]:
        """Get information about duplicates in a list of documents."""
        duplicates = []
        unique_count = 0
        
        for i, doc in enumerate(documents):
            doc_hash = self._generate_document_hash(doc)
            is_duplicate = doc_hash in self.document_hashes
            
            if is_duplicate:
                duplicates.append({
                    'index': i,
                    'hash': doc_hash,
                    'content_preview': doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
                })
            else:
                unique_count += 1
        
        return {
            'total_documents': len(documents),
            'unique_count': unique_count,
            'duplicate_count': len(duplicates),
            'duplicates': duplicates
        }

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with scores."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """Get retriever for the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        self.retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        return self.retriever

    def delete_store(self):
        """Delete the vector store from disk."""
        import shutil
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
            logger.info(f"Vector store deleted from: {self.store_path}")
            self.vector_store = None
            self.retriever = None
            self.document_hashes = set()
        else:
            logger.info(f"Store path does not exist: {self.store_path}")

    def clear_document_hashes(self):
        """Clear all document hashes (use with caution)."""
        self.document_hashes = set()
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
        logger.info("Document hashes cleared")

    def get_store_info(self):
        """Get information about the current vector store."""
        info = {
            "store_path": self.store_path,
            "embedding_model": self.embedding_model_name,
            "tracked_documents": len(self.document_hashes),
            "metadata_file": self.metadata_file
        }
        
        if self.vector_store is None:
            info["status"] = "No vector store initialized"
            info["index_size"] = 0
            info["dimension"] = "Unknown"
        else:
            info["status"] = "Vector store initialized"
            info["index_size"] = self.vector_store.index.ntotal
            info["dimension"] = self.vector_store.index.d if hasattr(self.vector_store.index, 'd') else "Unknown"
        
        logger.info("Vector Store Information:")
        for key, value in info.items():
            logger.info(f"   {key}: {value}")
        
        return info