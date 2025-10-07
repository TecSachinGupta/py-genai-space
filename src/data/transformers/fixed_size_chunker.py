from typing import Any, Dict, List
from src.schemas import TextChunker

class FixedSizeChunker(TextChunker):
    """Chunks text into fixed-size pieces with optional overlap"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space > 0:
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space
            
            chunk = {
                'text': chunk_text.strip(),
                'chunk_id': chunk_id,
                'start_pos': start,
                'end_pos': end,
                'chunk_size': len(chunk_text),
                'chunking_method': 'fixed_size',
                'metadata': metadata or {}
            }
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move start position considering overlap
            start = max(start + self.chunk_size - self.overlap, end)
            
            if start >= len(text):
                break
        
        return chunks