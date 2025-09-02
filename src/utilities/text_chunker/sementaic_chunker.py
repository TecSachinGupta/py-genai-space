from typing import Any, Dict, List
from src.schemas.text_chunker import TextChunker

import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SemanticChunker(TextChunker):
    """Chunks text based on semantic boundaries (paragraphs, sections)"""
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1500):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        chunk_id = 0
        
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_length = len(paragraph)
            
            # If single paragraph is too long, split it further
            if paragraph_length > self.max_chunk_size:
                # First, add current chunk if exists
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, chunk_id, start_pos, metadata
                    ))
                    chunk_id += 1
                    start_pos += len(chunk_text) + 2  # +2 for \n\n
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sentences = sent_tokenize(paragraph)
                temp_chunk = []
                temp_length = 0
                
                for sentence in sentences:
                    if temp_length + len(sentence) > self.max_chunk_size and temp_chunk:
                        chunk_text = ' '.join(temp_chunk)
                        chunks.append(self._create_chunk(
                            chunk_text, chunk_id, start_pos, metadata
                        ))
                        chunk_id += 1
                        start_pos += len(chunk_text) + 1
                        temp_chunk = []
                        temp_length = 0
                    
                    temp_chunk.append(sentence)
                    temp_length += len(sentence)
                
                if temp_chunk:
                    chunk_text = ' '.join(temp_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, chunk_id, start_pos, metadata
                    ))
                    chunk_id += 1
                    start_pos += len(chunk_text) + 1
            
            # Check if adding paragraph would exceed max size
            elif current_length + paragraph_length > self.max_chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text, chunk_id, start_pos, metadata
                ))
                chunk_id += 1
                start_pos += len(chunk_text) + 2
                current_chunk = [paragraph]
                current_length = paragraph_length
            
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        # Add remaining paragraphs
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text, chunk_id, start_pos, metadata
            ))
        
        # Filter out chunks that are too small
        chunks = [chunk for chunk in chunks if len(chunk['text']) >= self.min_chunk_size]
        
        return chunks
    
    def _create_chunk(self, text: str, chunk_id: int, start_pos: int, 
                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'text': text,
            'chunk_id': chunk_id,
            'start_pos': start_pos,
            'end_pos': start_pos + len(text),
            'chunk_size': len(text),
            'chunking_method': 'semantic',
            'metadata': metadata or {}
        }