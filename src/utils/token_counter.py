"""
Token Counter Utility

Since Gemini doesn't provide a native token counter ,
I implement a character-based approximation.

calcuation  = totalchars/4 = estimated token
example:
test :Thabo loves RPA
charCount = 15
calculatin = 15/4 = 3 3/15 = 3.2
"""

import logging
from typing import Optional


class TokenCounter:
    """
    Handles token counting for text content also considering the budget(bugdet management).
    """
    
    # Character to token ratio (conservative estimate)
    CHARS_PER_TOKEN = 4
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def count_tokens(self, text: str) -> int:
        """ 
        The approximation works as follows:
        - Average English word is 5 characters
        - Average token is 0.75 words (subword tokenization)
        - Therefore: 1 token ≈ 4 characters
        """
        if not text:
            return 0
        
        # Remove excessive whitespace for more accurate counting
        cleaned_text = ' '.join(text.split())
        char_count = len(cleaned_text)
        token_count = max(1, char_count // self.CHARS_PER_TOKEN)
        
        self.logger.debug(f"Counted {token_count} tokens from {char_count} characters")
        return token_count
    
    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """
        Count tokens for multiple texts.
        """
        return [self.count_tokens(text) for text in texts]
    
    def estimate_tokens_from_chars(self, char_count: int) -> int:
        """
        Estimate tokens from character count without processing text.
        """
        return max(1, char_count // self.CHARS_PER_TOKEN)
    
    def fits_budget(self, text: str, budget: int) -> bool:
        """
        Check if text fits within token budget.
        """
        token_count = self.count_tokens(text)
        return token_count <= budget
    
    def truncate_to_budget(self, text: str, budget: int) -> str:
        """
        Truncate text to fit within token budget.
        
        This is a simple truncation that preserves the beginning.
        """
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= budget:
            return text
        
        # Calculate approximate character limit
        target_chars = budget * self.CHARS_PER_TOKEN
        
        # Truncate and add ellipsis
        truncated = text[:target_chars].strip()
        
        # Try to break at last complete word
        last_space = truncated.rfind(' ')
        if last_space > target_chars * 0.8:  # At least 80% of target
            truncated = truncated[:last_space]
        
        truncated += "..."
        
        self.logger.info(f"Truncated text from {current_tokens} to ~{budget} tokens")
        return truncated


# Global singleton instance
_token_counter_instance: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """
    Get the global TokenCounter instance.
    Singleton TokenCounter instance is Returned
    """
    global _token_counter_instance
    if _token_counter_instance is None:
        _token_counter_instance = TokenCounter()
    return _token_counter_instance

