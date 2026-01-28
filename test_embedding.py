"""
Test embeddings in isolation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.retrieval.embeddings import get_embedding_generator

def test_embeddings():
    print("\n--- Testing Embeddings ---")
    
    generator = get_embedding_generator()
    
    # Test single embedding
    text = "Neural networks are machine learning models."
    print(f"\nGenerating embedding for: '{text}'")
    
    try:
        embedding = generator.generate_embedding(text)
        print(f"Success! Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Test batch
        texts = [
            "Machine learning is a subset of AI.",
            "Vector databases store embeddings.",
            "Semantic search finds similar meanings."
        ]
        
        print(f"\nGenerating {len(texts)} embeddings in batch...")
        embeddings = generator.generate_embeddings_batch(texts)
        print(f"Success! Generated {len(embeddings)} embeddings")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embeddings()


