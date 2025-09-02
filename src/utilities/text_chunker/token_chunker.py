from typing import Any, Dict, List
import tiktoken
from src.schemas.text_chunker import TextChunker

import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TokenChunker(TextChunker):
    """Chunks text based on token count (useful for LLM context limits)"""
    
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50, 
                 model: str = "gpt-3.5-turbo"):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.encoding_for_model(model)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Try to break at sentence boundary if not at end
            if end < len(tokens):
                sentences = sent_tokenize(chunk_text)
                if len(sentences) > 1:
                    # Keep all but last sentence to avoid cutting mid-sentence
                    chunk_text = ' '.join(sentences[:-1])
                    chunk_tokens = self.encoding.encode(chunk_text)
            
            chunk = {
                'text': chunk_text.strip(),
                'chunk_id': chunk_id,
                'token_count': len(chunk_tokens),
                'start_token': start,
                'end_token': start + len(chunk_tokens),
                'chunk_size': len(chunk_text),
                'chunking_method': 'token',
                'metadata': metadata or {}
            }
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move start considering overlap
            start = start + len(chunk_tokens) - self.overlap_tokens
            
            if start >= len(tokens):
                break
        
        return chunks