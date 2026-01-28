"""
Test the complete context assembly pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.context_assembler import get_context_assembler
from src.memory.convo_memory import get_conversation_memory
from src.memory.user_preferences import get_user_preferences
from src.tools.toolManager import get_tool_manager

def test_normal_assembly():
    print("\n" + "="*70)
    print("TEST 1: NORMAL ASSEMBLY (Within Budget)")
    print("="*70)
    
    assembler = get_context_assembler()
    
    # Assemble context for a simple query
    query = "How do neural networks learn through backpropagation?"
    
    assembled = assembler.assemble(query)
    
    # Print report
    print(assembler.get_assembly_report(assembled))
    
    # Show assembled context snippet
    full_context = assembled.get_full_context()
    print("\nASSEMBLED CONTEXT (first 500 chars):")
    print("-" * 70)
    print(full_context[:500] + "...")

def test_with_memory():
    print("\n" + "="*70)
    print("TEST 2: ASSEMBLY WITH CONVERSATION MEMORY")
    print("="*70)
    
    # Add conversation history
    memory = get_conversation_memory()
    memory.clear()
    
    memory.add_exchange(
        "What is machine learning?",
        "Machine learning is a subset of AI that enables computers to learn from data."
    )
    memory.add_exchange(
        "Can you give me an example?",
        "Sure! Spam detection is a classic example where ML learns to classify emails."
    )
    
    # Assemble with memory
    assembler = get_context_assembler()
    query = "How does it handle new types of spam?"
    
    assembled = assembler.assemble(query)
    
    print(assembler.get_assembly_report(assembled))

def test_budget_overflow():
    print("\n" + "="*70)
    print("TEST 3: BUDGET OVERFLOW HANDLING")
    print("="*70)
    
    # Create a very long query to trigger goal truncation
    long_query = """
    I'm working on a machine learning project where I need to understand 
    how neural networks work, specifically the backpropagation algorithm 
    and gradient descent optimization. I've read several tutorials but I'm 
    still confused about how the chain rule is applied during backprop.
    I also want to know about learning rates, batch sizes, epochs, 
    activation functions like ReLU and sigmoid, loss functions, 
    regularization techniques like dropout and L2 regularization,
    and how to prevent overfitting in deep neural networks.
    Additionally, I'm curious about different architectures like CNNs,
    RNNs, LSTMs, and Transformers. What is the best way to understand 
    all of these concepts intuitively and how do they relate to each other?
    """ * 5  # Repeat to make it very long
    
    # Add lots of memory
    memory = get_conversation_memory()
    memory.clear()
    
    for i in range(10):
        memory.add_exchange(
            f"Question {i} about neural networks and deep learning concepts",
            f"Answer {i} explaining various aspects of machine learning and AI systems"
        )
    
    # Add tool outputs
    tools = get_tool_manager()
    tools.clear()
    
    for i in range(5):
        tools.add_tool_output(
            "web_search",
            f"Search result {i}: Detailed information about neural networks, "
            f"backpropagation, gradient descent, and optimization algorithms. "
            f"This is a long result that contains lots of technical details." * 10,
            success=True
        )
    
    # Assemble - should trigger truncation
    assembler = get_context_assembler()
    assembled = assembler.assemble(long_query)
    
    print(assembler.get_assembly_report(assembled))
    
    # Show what got truncated
    if assembled.truncation_applied:
        print("\nTRUNCATION DETAILS:")
        for section, detail in assembled.truncation_details.items():
            print(f"  - {section}: {detail}")

if __name__ == "__main__":
    test_normal_assembly()
    test_with_memory()
    test_budget_overflow()
    
    print("\n" + "="*70)
    print("ALL CONTEXT ASSEMBLY TESTS COMPLETE")
    print("="*70)

