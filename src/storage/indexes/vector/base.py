from abc import ABC, abstractmethod
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabase(ABC):
    """Abstract base class for vector databases with enhanced functionality"""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], check_duplicates: bool = True) -> List[str]:
        """Add documents to the database and return their IDs"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        pass
    
    @abstractmethod
    def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update a document"""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        pass
    
    @abstractmethod
    def save_store(self, filepath: Optional[str] = None) -> bool:
        """Save the database to disk"""
        pass
    
    @abstractmethod
    def load_store(self, filepath: str) -> bool:
        """Load the database from disk"""
        pass
    
    def _generate_document_hash(self, document: Dict[str, Any]) -> str:
        """Generate a unique hash for a document based on its content and metadata."""
        # Create a string representation of the document
        content = document.get('text', '')
        metadata = document.get('metadata', {})
        
        # Sort metadata keys to ensure consistent hashing
        metadata_str = json.dumps(metadata, sort_keys=True, default=str)
        combined = f"{content}||{metadata_str}"
        
        # Generate SHA-256 hash
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def _filter_duplicate_documents(self, documents: List[Dict[str, Any]], 
                                   existing_hashes: Set[str]) -> Tuple[List[Dict[str, Any]], int]:
        """Filter out duplicate documents and return unique ones along with duplicate count."""
        unique_documents = []
        duplicate_count = 0
        
        for doc in documents:
            doc_hash = self._generate_document_hash(doc)
            
            if doc_hash not in existing_hashes:
                unique_documents.append(doc)
                existing_hashes.add(doc_hash)
            else:
                duplicate_count += 1
                logger.debug(f"Skipping duplicate document with hash: {doc_hash[:16]}...")
        
        return unique_documents, duplicate_count