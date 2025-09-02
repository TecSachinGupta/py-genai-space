from typing import Any, Dict, List
from src.schemas.text_chunker import TextChunker

import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentenceChunker(TextChunker):
    """Chunks text by sentences, grouping them into larger chunks"""
    
    def __init__(self, max_sentences: int = 5, max_chars: int = 1000):
        self.max_sentences = max_sentences
        self.max_chars = max_chars
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        chunks = []
        chunk_id = 0
        
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed limits
            if (len(current_chunk) >= self.max_sentences or 
                current_length + sentence_length > self.max_chars) and current_chunk:
                
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                
                chunk = {
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'sentence_count': len(current_chunk),
                    'chunk_size': len(chunk_text),
                    'chunking_method': 'sentence',
                    'metadata': metadata or {}
                }
                
                chunks.append(chunk)
                chunk_id += 1
                
                # Reset for next chunk
                start_pos = end_pos + 1
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            
            chunk = {
                'text': chunk_text,
                'chunk_id': chunk_id,
                'start_pos': start_pos,
                'end_pos': start_pos + len(chunk_text),
                'sentence_count': len(current_chunk),
                'chunk_size': len(chunk_text),
                'chunking_method': 'sentence',
                'metadata': metadata or {}
            }
            
            chunks.append(chunk)
        
        return chunks