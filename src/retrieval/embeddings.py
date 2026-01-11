"""
Embeddings Module

Handles text-to-vector conversion using Google's embedding models.
Embeddings allow us to find semantically similar documents even when
they don't share exact keywords.
"""

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

from google import genai


class EmbeddingGenerator:
    """
    Generates embeddings using Google Gemini's embedding model.
    
    Embeddings are vector representations of text that capture semantic meaning.
    Similar texts will have similar vectors (measured by cosine similarity).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            api_key: Google API key
            model_name: Embedding model to use
        """
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        raw_model_name = model_name or os.getenv(
            'GEMINI_EMBEDDING_MODEL',
            'text-embedding-004'
        )
        
        if not raw_model_name.startswith('models/'):
            self.model_name = f'models/{raw_model_name}'
        else:
            self.model_name = raw_model_name
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        self.client = genai.Client(api_key=self.api_key)
        
        self.logger.info(f"Initialized embeddings with model: {self.model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding (list of floats)
            
        Raises:
            Exception: If embedding generation fails
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return [0.0] * 768
        
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            
            # The response is a complex object, we need to extract the actual values
            # Response structure: response.embeddings[0].values
            if hasattr(response, 'embeddings') and len(response.embeddings) > 0:
                embedding_obj = response.embeddings[0]
                if hasattr(embedding_obj, 'values'):
                    embedding = list(embedding_obj.values)
                else:
                    embedding = list(embedding_obj)
            elif isinstance(response, dict):
                # Handle dict response
                if 'embeddings' in response:
                    embedding = list(response['embeddings'][0]['values'])
                elif 'embedding' in response:
                    embedding = list(response['embedding'])
                else:
                    raise ValueError(f"Unexpected response structure: {type(response)}")
            else:
                # Last resort: try to iterate
                embedding = list(response)
            
            self.logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated {i + 1}/{len(texts)} embeddings")
                    
            except Exception as e:
                self.logger.error(f"Failed to embed text {i}: {str(e)}")
                embeddings.append([0.0] * 768)
        
        self.logger.info(f"Completed batch embedding: {len(embeddings)} vectors")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.
        
        Returns:
            Embedding dimension (typically 768 for text-embedding-004)
        """
        return 768


_embedding_generator_instance: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get the global EmbeddingGenerator instance.
    
    Returns:
        Singleton EmbeddingGenerator instance
    """
    global _embedding_generator_instance
    if _embedding_generator_instance is None:
        _embedding_generator_instance = EmbeddingGenerator()
    return _embedding_generator_instance

