import time
from typing import List
import numpy as np
import openai
from src.schemas.embedding_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536 if model == "text-embedding-ada-002" else 1536
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts with batch processing"""
        if not texts:
            return np.array([])
        
        # OpenAI has limits on batch size, so we process in chunks
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                # Return zeros for failed batch
                batch_embeddings = [[0.0] * self._dimension] * len(batch_texts)
                all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """Embed a single text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error embedding text: {str(e)}")
            return np.zeros(self._dimension)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self.model