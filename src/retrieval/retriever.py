"""
Retriever - High-level search interface

Orchestrates the retrieval process:
1. Search vector store for similar documents
2. Rank results by relevance-important 
3. Format for context assembly
"""

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

from src.retrieval.vector_store import get_vector_store
from src.core.prioritizer import RetrievalChunk
from src.utils.token_counter import get_token_counter


class Retriever:
    """
    High-level interface for document retrieval.
    
    Handles the complete retrieval pipeline from query to
    formatted, budget-compliant results.
    """
    
    def __init__(
        self,
        max_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ):
        """
        Initialize retriever.
        
        Args:
            max_results: Maximum documents to retrieve
            similarity_threshold: Minimum similarity score (0-1)
        """
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        
        self.max_results = max_results or int(
            os.getenv('MAX_RETRIEVAL_RESULTS', '10')
        )
        self.similarity_threshold = similarity_threshold or float(
            os.getenv('SIMILARITY_THRESHOLD', '0.7')
        )
        
        self.vector_store = get_vector_store()
        self.token_counter = get_token_counter()
        
        self.logger.info(
            f"Initialized Retriever (max_results={self.max_results}, "
            f"threshold={self.similarity_threshold})"
        )
    
    def retrieve(self, query: str) -> List[RetrievalChunk]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of RetrievalChunk objects, sorted by relevance
        """
        # Search vector store
        results = self.vector_store.search(query, n_results=self.max_results)
        
        # Filter by threshold and convert to RetrievalChunk
        chunks = []
        for result in results:
            if result['score'] >= self.similarity_threshold:
                chunk = RetrievalChunk(
                    content=result['content'],
                    score=result['score'],
                    source=result['metadata'].get('filename', result['id']),
                    tokens=self.token_counter.count_tokens(result['content'])
                )
                chunks.append(chunk)
        
        self.logger.info(
            f"Retrieved {len(chunks)} chunks above threshold "
            f"(from {len(results)} total results)"
        )
        
        return chunks
    
    def retrieve_formatted(
        self,
        query: str,
        budget: int
    ) -> str:
        """
        Retrieve and format results within token budget.
        
        This is the main method used by context assembly.
        
        Args:
            query: Search query
            budget: Token budget for retrieval section
            
        Returns:
            Formatted retrieval string within budget
        """
        from src.core.prioritizer import get_prioritizer
        
        # Retrieve chunks
        chunks = self.retrieve(query)
        
        if not chunks:
            self.logger.warning("No relevant documents found")
            return ""
        
        # Truncate to budget using prioritizer
        prioritizer = get_prioritizer()
        formatted = prioritizer.truncate_retrieval(chunks, budget)
        
        final_tokens = self.token_counter.count_tokens(formatted)
        self.logger.info(f"Formatted retrieval: {final_tokens} tokens")
        
        return formatted


# Global singleton instance
_retriever_instance: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """
    Get the global Retriever instance.
    
    Returns:
        Singleton Retriever instance
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance

