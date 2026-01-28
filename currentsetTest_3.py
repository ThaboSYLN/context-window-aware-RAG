"""
Test the complete retrieval pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.retrieval.vector_store import get_vector_store
from src.retrieval.retriever import get_retriever

def initialize_corpus():
    """Load sample documents into vector store"""
    print("\n--- Initializing Corpus ---")
    
    vector_store = get_vector_store()
    
    # Clear existing data
    stats = vector_store.get_collection_stats()
    if stats['document_count'] > 0:
        print(f"Collection already has {stats['document_count']} documents")
        return
    
    # Load from corpus directory
    loaded = vector_store.load_corpus_from_directory('./data/corpus')
    print(f"Loaded {loaded} documents")

def test_search():
    """Test semantic search"""
    print("\n--- Testing Semantic Search ---")
    
    retriever = get_retriever()
    
    queries = [
        "How do neural networks learn?",
        "What is vector search?",
        "Explain NLP techniques"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        # Get formatted results within budget
        results = retriever.retrieve_formatted(query, budget=200)
        
        if results:
            print(results[:300] + "..." if len(results) > 300 else results)
        else:
            print("No results found")

if __name__ == "__main__":
    initialize_corpus()
    test_search()
    print("\n--- Retrieval test complete ---")

