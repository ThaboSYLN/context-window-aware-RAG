"""
Test basic setup of utils and LLM client
"""

import sys
from pathlib import Path

# Add src to path
#sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.token_counter import get_token_counter
from src.utils.logger import setup_logger
from src.llm.client import get_gemini_client

def test_token_counter():
    print("\n--- Testing Token Counter ---")
    counter = get_token_counter()
    
    text = "This is a test sentence to count tokens."
    tokens = counter.count_tokens(text)
    print(f"Text: '{text}'")
    print(f"Estimated tokens: {tokens}")
    print(f"Fits in 50 token budget: {counter.fits_budget(text, 50)}")

def test_logger():
    print("\n--- Testing Logger ---")
    logger = setup_logger('test_logger', level='INFO')
    logger.info("Logger is working correctly")
    logger.debug("This debug message won't show at INFO level")

def test_gemini_client():
    print("\n--- Testing Gemini Client ---")
    try:
        client = get_gemini_client()
        print(f"Client initialized: {client.get_model_info()}")
        
        response = client.generate("Say hello in one sentence.", temperature=0.7)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your GEMINI_API_KEY is set in .env file")

if __name__ == "__main__":
    test_token_counter()
    test_logger()
    test_gemini_client()
    print("\n--- All basic tests complete ---")

