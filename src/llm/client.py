"""
LLM Client for Google Gemini

Provides a clean interface to Google's Gemini API.
Handles configuration, request formatting, and response parsing.

Updated to use the new google-genai library.
"""

import os
from typing import Optional, Dict, Any
import logging

from google import genai
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv


class GeminiClient:
    """
    Client for interacting with Google Gemini API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize Gemini client.
        """
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        
        # Get configuration from environment or parameters
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name or os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in .env file or pass as parameter."
            )
        
        # Configure Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        self.logger.info(f"Initialized Gemini client with model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from Gemini.
        """
        try:
            self.logger.debug(f"Generating response with temperature={temperature}")
            
            # Configure generation parameters
            config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # Extract text from response
            if not response.text:
                self.logger.warning("Empty response from Gemini")
                return "I apologize, but I couldn't generate a response."
            
            self.logger.info("Successfully generated response")
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_with_context(
        self,
        context_parts: Dict[str, str],
        temperature: float = 0.7
    ) -> str:
        """
        Generate response given structured context parts.
        
        This method takes the assembled context sections and
        formats them into a proper prompt.
        
        Args:
            context_parts: Dictionary with keys like 'instructions',
                          'goal', 'memory', 'retrieval', 'tool_outputs'
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        # Assemble the complete prompt
        prompt_parts = []
        
        if context_parts.get('instructions'):
            prompt_parts.append(f"INSTRUCTIONS:\n{context_parts['instructions']}\n")
        
        if context_parts.get('memory'):
            prompt_parts.append(f"CONVERSATION CONTEXT:\n{context_parts['memory']}\n")
        
        if context_parts.get('retrieval'):
            prompt_parts.append(f"RELEVANT KNOWLEDGE:\n{context_parts['retrieval']}\n")
        
        if context_parts.get('tool_outputs'):
            prompt_parts.append(f"RECENT ACTIONS:\n{context_parts['tool_outputs']}\n")
        
        if context_parts.get('goal'):
            prompt_parts.append(f"USER QUERY:\n{context_parts['goal']}")
        
        prompt = "\n".join(prompt_parts)
        
        self.logger.debug(f"Assembled prompt with {len(prompt_parts)} sections")
        
        return self.generate(prompt, temperature=temperature)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'provider': 'Google Gemini'
        }


# Global client instance
_gemini_client_instance: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """
    Get the global GeminiClient instance.
    
    Returns:
        Singleton GeminiClient instance
    """
    global _gemini_client_instance
    if _gemini_client_instance is None:
        _gemini_client_instance = GeminiClient()
    return _gemini_client_instance

