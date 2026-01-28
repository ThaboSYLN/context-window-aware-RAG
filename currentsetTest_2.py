"""
Test budget management and prioritization
"""

import sys
from pathlib import Path

#sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.budget_manager import get_budget_manager, BudgetAllocation
from src.core.prioritizer import get_prioritizer, RetrievalChunk

def test_budget_manager():
    print("\n" + "="*60)
    print("TESTING BUDGET MANAGER")
    print("="*60)
    
    manager = get_budget_manager()
    
    # Test context within budget
    context = {
        'instructions': 'You are a helpful assistant.' * 10,  # ~50 tokens
        'goal': 'Explain neural networks.',  # ~5 tokens
        'memory': 'Previous: user asked about ML',  # ~10 tokens
        'retrieval': 'Neural networks are...' * 20,  # ~100 tokens
        'tool_outputs': 'Searched for: ML basics'  # ~10 tokens
    }
    
    valid, allocation, overages = manager.validate_context(context)
    print(f"\nValidation result: {'PASS' if valid else 'FAIL'}")
    print(f"Allocation: {allocation.to_dict()}")
    if overages:
        print(f"Overages: {overages}")
    
    print("\n" + manager.format_budget_report(allocation))

def test_prioritizer():
    print("\n" + "="*60)
    print("TESTING PRIORITIZER")
    print("="*60)
    
    prioritizer = get_prioritizer()
    
    # Test goal truncation
    long_goal = (
        "I'm working on a machine learning project where I need to understand "
        "how neural networks work, specifically the backpropagation algorithm "
        "and gradient descent optimization. I've read several tutorials but I'm "
        "still confused about how the chain rule is applied during backprop. "
        "What is the best way to understand backpropagation intuitively beacuse I live it ?"
    )
    
    print("\n--- Goal Truncation Test ---")
    print(f"Original ({prioritizer.token_counter.count_tokens(long_goal)} tokens):")
    print(long_goal)
    
    truncated_goal = prioritizer.truncate_goal(long_goal, 30)
    print(f"\nTruncated to ~30 tokens ({prioritizer.token_counter.count_tokens(truncated_goal)} tokens):")
    print(truncated_goal)
    
    # Test retrieval truncation
    print("\n--- Retrieval Truncation Test ---")
    chunks = [
        RetrievalChunk("Neural networks basics...", 0.95, "doc1.txt", 20),
        RetrievalChunk("Backpropagation explained...", 0.89, "doc2.txt", 25),
        RetrievalChunk("History of AI...", 0.45, "doc3.txt", 30),
        RetrievalChunk("Deep learning intro...", 0.78, "doc4.txt", 22),
    ]
    
    truncated_retrieval = prioritizer.truncate_retrieval(chunks, 50)
    print("Truncated retrieval (budget: 50 tokens):")
    print(truncated_retrieval)

if __name__ == "__main__":
    test_budget_manager()
    test_prioritizer()
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)

