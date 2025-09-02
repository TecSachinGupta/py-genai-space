import numpy as np
import json
import sqlite3
import os
import logging
from typing import List, Dict, Any, Optional, Set
import uuid

import faiss
from src.schemas.vector_database import VectorDatabase

logger = logging.getLogger(__name__)

class FAISSVectorDB(VectorDatabase):
    """FAISS-based vector database with duplicate prevention and better metadata management"""
    
    def __init__(self, dimension: int, index_type: str = "flat", 
                 store_path: str = "./faiss_vector_store", 
                 db_path: Optional[str] = None):
        self.dimension = dimension
        self.store_path = store_path
        self.index_type = index_type
        
        # Create store directory
        os.makedirs(store_path, exist_ok=True)
        
        # SQLite path
        if db_path is None:
            db_path = os.path.join(store_path, "faiss_metadata.sqlite")
        self.db_path = db_path
        
        # Metadata file for document hashes
        self.metadata_file = os.path.join(store_path, "document_metadata.json")
        self.document_hashes: Set[str] = set()
        
        # Initialize FAISS index
        self._init_faiss_index()
        
        # Initialize SQLite for metadata
        self._init_sqlite()
        
        # Load existing document hashes
        self._load_document_hashes()
        
        logger.info(f"Enhanced FAISS Vector DB initialized at {store_path}")
    
    def _init_faiss_index(self):
        """Initialize FAISS index based on type"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)
        elif self.index_type == "ivf":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "hnsw":
            # HNSW index for fast approximate search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def _init_sqlite(self):
        """Initialize SQLite database for metadata"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    metadata TEXT,
                    embedding_dimension INTEGER,
                    document_hash TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index on document_hash for faster lookups
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_hash ON documents(document_hash)
            """)
            
            self.conn.commit()
            logger.info("SQLite metadata database initialized")
        except Exception as e:
            logger.error(f"Error initializing SQLite: {e}")
            raise e
    
    def _load_document_hashes(self):
        """Load document hashes from metadata file and database."""
        # Load from JSON file
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.document_hashes = set(data.get('document_hashes', []))
                logger.info(f"Loaded {len(self.document_hashes)} document hashes from metadata file")
            except Exception as e:
                logger.warning(f"Error loading document metadata file: {e}")
        
        # Also load from database as backup
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT document_hash FROM documents WHERE document_hash IS NOT NULL")
            db_hashes = {row[0] for row in cursor.fetchall()}
            self.document_hashes.update(db_hashes)
            logger.info(f"Total document hashes loaded: {len(self.document_hashes)}")
        except Exception as e:
            logger.warning(f"Error loading hashes from database: {e}")
    
    def _save_document_hashes(self):
        """Save document hashes to metadata file."""
        try:
            metadata = {
                'document_hashes': list(self.document_hashes),
                'total_documents': len(self.document_hashes),
                'dimension': self.dimension,
                'index_type': self.index_type
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved {len(self.document_hashes)} document hashes to metadata")
        except Exception as e:
            logger.error(f"Error saving document metadata: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]], check_duplicates: bool = True) -> List[str]:
        """Add documents to FAISS index and SQLite with optional duplicate checking"""
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
        
        try:
            for doc in documents:
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                # Extract embedding and normalize for cosine similarity
                embedding = doc['embedding']
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                # Normalize for cosine similarity with inner product
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings.append(embedding)
                
                # Generate document hash
                doc_hash = self._generate_document_hash(doc)
                
                # Store metadata in SQLite
                metadata = doc.copy()
                del metadata['embedding']  # Don't store embedding twice
                
                self.conn.execute(
                    """INSERT INTO documents 
                       (id, text, metadata, embedding_dimension, document_hash) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (doc_id, doc.get('text', ''), json.dumps(metadata), 
                     self.dimension, doc_hash)
                )
                
                # Add to hash set
                if check_duplicates:
                    self.document_hashes.add(doc_hash)
            
            # Add embeddings to FAISS index
            if embeddings:
                embeddings_matrix = np.array(embeddings).astype(np.float32)
                self.index.add(embeddings_matrix)
            
            self.conn.commit()
            
            if check_duplicates:
                self._save_document_hashes()
            
            logger.info(f"Added {len(doc_ids)} documents to FAISS vector database")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            self.conn.rollback()
            raise e
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using FAISS"""
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no documents to search")
            return []
        
        try:
            # Normalize query embedding for cosine similarity
            query_embedding = query_embedding.astype(np.float32)
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Get metadata from SQLite
            results = []
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, text, metadata, document_hash FROM documents ORDER BY rowid")
            rows = cursor.fetchall()
            
            for i, idx in enumerate(indices[0]):
                if idx < len(rows) and idx >= 0:
                    doc_id, text, metadata_json, doc_hash = rows[idx]
                    
                    try:
                        metadata = json.loads(metadata_json)
                    except:
                        metadata = {}
                    
                    result = {
                        'id': doc_id,
                        'text': text,
                        'metadata': metadata,
                        'score': float(scores[0][i]),
                        'document_hash': doc_hash
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document metadata (Note: FAISS doesn't support efficient individual deletion)"""
        try:
            # Get document hash before deletion
            cursor = self.conn.cursor()
            cursor.execute("SELECT document_hash FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if row:
                doc_hash = row[0]
                # Remove from database
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                deleted = cursor.rowcount > 0
                self.conn.commit()
                
                # Remove from hash set
                if doc_hash and doc_hash in self.document_hashes:
                    self.document_hashes.remove(doc_hash)
                    self._save_document_hashes()
                
                if deleted:
                    logger.warning(f"Document {doc_id} metadata deleted, but FAISS index not rebuilt. "
                                 "Consider rebuilding index for optimal performance.")
                
                return deleted
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, text, metadata, document_hash FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if row:
                doc_id, text, metadata_json, doc_hash = row
                try:
                    metadata = json.loads(metadata_json)
                except:
                    metadata = {}
                
                return {
                    'id': doc_id,
                    'text': text,
                    'metadata': metadata,
                    'document_hash': doc_hash
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update document metadata (embedding updates require rebuild)"""
        try:
            metadata = document.copy()
            if 'embedding' in metadata:
                del metadata['embedding']
            
            cursor = self.conn.cursor()
            cursor.execute(
                """UPDATE documents 
                   SET text = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP 
                   WHERE id = ?""",
                (document.get('text', ''), json.dumps(metadata), doc_id)
            )
            
            updated = cursor.rowcount > 0
            self.conn.commit()
            
            if updated:
                logger.info(f"Document {doc_id} updated successfully")
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            return {
                'total_documents': total_docs,
                'faiss_index_size': self.index.ntotal,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'tracked_hashes': len(self.document_hashes),
                'store_path': self.store_path
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def save_store(self, filepath: Optional[str] = None) -> bool:
        """Save FAISS index and metadata to disk"""
        try:
            save_path = filepath or self.store_path
            index_path = os.path.join(save_path, "faiss.index")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save document hashes
            self._save_document_hashes()
            
            logger.info(f"FAISS store saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving store: {e}")
            return False
    
    def load_store(self, filepath: str) -> bool:
        """Load FAISS index from disk"""
        try:
            index_path = os.path.join(filepath, "faiss.index")
            
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                self.store_path = filepath
                
                # Reload document hashes
                self.metadata_file = os.path.join(filepath, "document_metadata.json")
                self._load_document_hashes()
                
                logger.info(f"FAISS store loaded from {filepath}")
                return True
            else:
                logger.warning(f"Index file not found at {index_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading store from {filepath}: {e}")
            return False