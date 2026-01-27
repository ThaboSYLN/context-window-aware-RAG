"""
Token Counter Utility

Provides accurate token counting using tiktoken library.
Falls back to character-based approximation if tiktoken unavailable.

Token counting accuracy is CRITICAL for budget enforcement.
"""

import logging
from typing import Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class TokenCounter:
    """
    Handles token counting for text content with budget management.
    Uses tiktoken for accurate counting when available.
    """
    
    # Character to token ratio (fallback only)
    CHARS_PER_TOKEN = 4
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for tiktoken encoding (default: gpt-3.5-turbo)
        """
        self.logger = logging.getLogger(__name__)
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
                self.use_tiktoken = True
                self.logger.info(f"Using tiktoken with {model} encoding for accurate token counting")
            except Exception as e:
                self.logger.warning(f"Failed to load tiktoken encoding: {e}. Using fallback.")
                self.use_tiktoken = False
        else:
            self.logger.warning("tiktoken not available. Using character-based approximation. Install with: pip install tiktoken")
            self.use_tiktoken = False
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Uses tiktoken for accuracy when available, otherwise approximates.
        
        Args:
            text: Text to count
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        if self.use_tiktoken:
            try:
                tokens = self.encoding.encode(text)
                token_count = len(tokens)
                self.logger.debug(f"Counted {token_count} tokens (tiktoken)")
                return token_count
            except Exception as e:
                self.logger.error(f"Tiktoken encoding failed: {e}. Using fallback.")
                # Fall through to approximation
        
        # Fallback: character-based approximation
        cleaned_text = ' '.join(text.split())
        char_count = len(cleaned_text)
        token_count = max(1, char_count // self.CHARS_PER_TOKEN)
        
        self.logger.debug(f"Counted {token_count} tokens from {char_count} characters (approximation)")
        return token_count
    
    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """
        Count tokens for multiple texts efficiently.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of token counts
        """
        return [self.count_tokens(text) for text in texts]
    
    def estimate_tokens_from_chars(self, char_count: int) -> int:
        """
        Estimate tokens from character count without processing text.
        
        Args:
            char_count: Number of characters
            
        Returns:
            Estimated token count
        """
        return max(1, char_count // self.CHARS_PER_TOKEN)
    
    def fits_budget(self, text: str, budget: int) -> bool:
        """
        Check if text fits within token budget.
        
        Args:
            text: Text to check
            budget: Token budget
            
        Returns:
            True if text fits within budget
        """
        token_count = self.count_tokens(text)
        fits = token_count <= budget
        
        if not fits:
            self.logger.debug(f"Text exceeds budget: {token_count} > {budget}")
        
        return fits
    
    def truncate_to_budget(self, text: str, budget: int) -> str:
        """
        Truncate text to fit within token budget.
        
        Preserves the beginning and tries to break at word boundaries.
        
        Args:
            text: Text to truncate
            budget: Token budget
            
        Returns:
            Truncated text within budget
        """
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= budget:
            return text
        
        self.logger.info(f"Truncating text from {current_tokens} to {budget} tokens")
        
        # Binary search for optimal truncation point
        left, right = 0, len(text)
        best_truncation = ""
        
        while left < right:
            mid = (left + right + 1) // 2
            candidate = text[:mid]
            
            if self.count_tokens(candidate) <= budget - 3:  # Reserve 3 tokens for "..."
                best_truncation = candidate
                left = mid
            else:
                right = mid - 1
        
        # Try to break at last complete word
        truncated = best_truncation.strip()
        last_space = truncated.rfind(' ')
        
        if last_space > len(truncated) * 0.8:  # At least 80% of target
            truncated = truncated[:last_space]
        
        truncated += "..."
        
        final_tokens = self.count_tokens(truncated)
        self.logger.debug(f"Truncation result: {final_tokens} tokens")
        
        return truncated


# Global singleton instance
_token_counter_instance: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """
    Get the global TokenCounter instance.
    
    Returns:
        Singleton TokenCounter instance
    """
    global _token_counter_instance
    if _token_counter_instance is None:
        _token_counter_instance = TokenCounter()
    return _token_counter_instance

