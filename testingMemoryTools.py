"""
Test memory and tool management
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.memory.convo_memory import get_conversation_memory
from src.memory.user_preferences import get_user_preferences
from src.tools.toolManager import get_tool_manager
from src.core.prioritizer import get_prioritizer

def test_conversation_memory():
    print("\n" + "="*60)
    print("TESTING CONVERSATION MEMORY")
    print("="*60)
    
    memory = get_conversation_memory()
    
    # Add some exchanges
    memory.add_exchange(
        "What is machine learning?",
        "Machine learning is a subset of AI that enables computers to learn from data."
    )
    
    memory.add_exchange(
        "How does it work?",
        "It uses algorithms to identify patterns in data and make predictions."
    )
    
    # Format for context
    formatted = memory.format_for_context()
    print("\nFormatted Memory:")
    print(formatted)
    
    # Get as items for prioritization
    items = memory.get_memory_items()
    print(f"\nMemory items: {len(items)}")
    
    # Test truncation
    prioritizer = get_prioritizer()
    truncated = prioritizer.truncate_memory(items, budget=30)
    print(f"\nTruncated to 30 tokens:")
    print(truncated)

def test_user_preferences():
    print("\n" + "="*60)
    print("TESTING USER PREFERENCES")
    print("="*60)
    
    prefs = get_user_preferences()
    
    # Set some preferences
    prefs.set_preference("expertise_level", "intermediate")
    prefs.set_preference("domain", "machine learning")
    
    # Format for context
    formatted = prefs.format_for_context()
    print(f"\nFormatted Preferences:\n{formatted}")

def test_tool_manager():
    print("\n" + "="*60)
    print("TESTING TOOL MANAGER")
    print("="*60)
    
    tools = get_tool_manager()
    
    # Simulate some tool executions
    tools.simulate_search(
        "neural networks",
        [
            "Neural networks are inspired by biological neurons",
            "They consist of layers of interconnected nodes"
        ]
    )
    
    tools.add_tool_output(
        "calculator",
        "Result: 42",
        success=True
    )
    
    # Format for context
    formatted = tools.format_for_context()
    print("\nFormatted Tool Outputs:")
    print(formatted[:200] + "..." if len(formatted) > 200 else formatted)
    
    # Get stats
    stats = tools.get_stats()
    print(f"\nTool Stats: {stats}")

if __name__ == "__main__":
    test_conversation_memory()
    test_user_preferences()
    test_tool_manager()
    print("\n" + "="*60)
    print("ALL MEMORY AND TOOL TESTS COMPLETE")
    print("="*60)

