"""
Enhanced Retriever with Web Scraping Integration

Orchestrates retrieval from BOTH sources:
1. Local corpus (vector database)
2. Web scraping (dynamic, cached)

Budget Management:
- STRICT 550 token limit
- Intelligent source mixing based on relevance
- Web scraping triggered automatically when beneficial

Strategy:
- First: Check local corpus
- Then: Determine if web search would help
- Combine results and rank by relevance
- Fit within budget (prioritize highest scores)
"""

import os
import logging
from typing import List, Optional, Dict
from dotenv import load_dotenv

from src.retrieval.vector_store import get_vector_store
from src.retrieval.web_scrapper import get_web_scraper
from src.retrieval.embeddings import get_embedding_generator
from src.core.prioritizer import RetrievalChunk
from src.utils.token_counter import get_token_counter


class EnhancedRetriever:
    """
    Advanced retriever that combines local corpus and web scraping.
    
    Automatically determines when to scrape web based on:
    - Local corpus relevance scores
    - Query characteristics
    - Cache availability
    
    Maintains STRICT 550 token budget across all sources.
    """
    
    def __init__(
        self,
        max_corpus_results: Optional[int] = None,
        max_web_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        web_trigger_threshold: Optional[float] = None
    ):
        """
        Initialize enhanced retriever.
        
        Args:
            max_corpus_results: Maximum results from local corpus
            max_web_results: Maximum results from web
            similarity_threshold: Minimum similarity score for results
            web_trigger_threshold: If best corpus score below this, trigger web search
        """
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        
        self.max_corpus_results = max_corpus_results or int(
            os.getenv('MAX_RETRIEVAL_RESULTS', '10')
        )
        self.max_web_results = max_web_results or 3  # Fewer web results (expensive)
        
        self.similarity_threshold = similarity_threshold or float(
            os.getenv('SIMILARITY_THRESHOLD', '0.3')
        )
        self.web_trigger_threshold = web_trigger_threshold or float(
            os.getenv('WEB_TRIGGER_THRESHOLD', '0.5')
        )
        
        # Initialize components
        self.vector_store = get_vector_store()
        self.web_scraper = get_web_scraper()
        self.embedding_generator = get_embedding_generator()
        self.token_counter = get_token_counter()
        
        self.logger.info(
            f"Initialized EnhancedRetriever "
            f"(corpus={self.max_corpus_results}, web={self.max_web_results}, "
            f"threshold={self.similarity_threshold}, web_trigger={self.web_trigger_threshold})"
        )
    
    def _should_use_web(self, corpus_results: List[Dict], query: str) -> bool:
        """
        Determine if web scraping would be beneficial.
        
        Triggers web search if:
        1. No corpus results OR
        2. Best corpus result below web_trigger_threshold OR
        3. Query contains "recent", "latest", "current", "today"
        
        Args:
            corpus_results: Results from local corpus
            query: User query
            
        Returns:
            True if web search should be performed
        """
        # No corpus results? Definitely use web
        if not corpus_results:
            self.logger.info("No corpus results - triggering web search")
            return True
        
        # Check best corpus score
        best_score = max(r['score'] for r in corpus_results)
        if best_score < self.web_trigger_threshold:
            self.logger.info(
                f"Best corpus score ({best_score:.3f}) below threshold "
                f"({self.web_trigger_threshold:.3f}) - triggering web search"
            )
            return True
        
        # Check for recency indicators
        recency_keywords = ['recent', 'latest', 'current', 'today', 'new', '2024', '2025', '2026']
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in recency_keywords):
            self.logger.info("Query contains recency indicators - triggering web search")
            return True
        
        self.logger.info(f"Corpus results sufficient (best score: {best_score:.3f}) - skipping web search")
        return False
    
    def _scrape_and_embed_web_results(self, query: str) -> List[RetrievalChunk]:
        """
        Scrape web and convert to RetrievalChunks.
        
        Args:
            query: Search query
            
        Returns:
            List of RetrievalChunk objects from web
        """
        try:
            # Scrape web (uses cache automatically)
            web_results = self.web_scraper.scrape_for_query(query)
            
            if not web_results:
                self.logger.warning("No web results found")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Convert to chunks with similarity scores
            chunks = []
            
            for result in web_results:
                content = result['content']
                
                # Generate embedding for content
                content_embedding = self.embedding_generator.generate_embedding(content)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, content_embedding)
                
                # Create chunk
                chunk = RetrievalChunk(
                    content=f"[Web: {result['title']}]\n{content}",
                    score=similarity,
                    source=f"web:{result['url']}",
                    tokens=self.token_counter.count_tokens(content)
                )
                
                chunks.append(chunk)
            
            self.logger.info(f"Created {len(chunks)} chunks from web results")
            return chunks
        
        except Exception as e:
            self.logger.error(f"Failed to process web results: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def retrieve(self, query: str) -> List[RetrievalChunk]:
        """
        Retrieve relevant content from corpus AND web.
        
        This method:
        1. Retrieves from local corpus
        2. Determines if web search needed
        3. Optionally scrapes web
        4. Combines and ranks all results
        5. Returns chunks sorted by relevance
        
        Args:
            query: Search query
            
        Returns:
            List of RetrievalChunk objects, sorted by relevance
        """
        all_chunks = []
        
        # Step 1: Retrieve from local corpus
        self.logger.info(f"Retrieving from local corpus for: {query}")
        
        corpus_results = self.vector_store.search(query, n_results=self.max_corpus_results)
        
        # Convert to RetrievalChunks
        for result in corpus_results:
            if result['score'] >= self.similarity_threshold:
                chunk = RetrievalChunk(
                    content=result['content'],
                    score=result['score'],
                    source=f"corpus:{result['metadata'].get('filename', result['id'])}",
                    tokens=self.token_counter.count_tokens(result['content'])
                )
                all_chunks.append(chunk)
        
        self.logger.info(
            f"Retrieved {len(all_chunks)} chunks from corpus "
            f"(above threshold {self.similarity_threshold})"
        )
        
        # Step 2: Determine if web search needed
        if self._should_use_web(corpus_results, query):
            # Step 3: Scrape web
            self.logger.info("Performing web search...")
            web_chunks = self._scrape_and_embed_web_results(query)
            
            # Filter by threshold
            web_chunks = [c for c in web_chunks if c.score >= self.similarity_threshold]
            
            self.logger.info(
                f"Retrieved {len(web_chunks)} chunks from web "
                f"(above threshold {self.similarity_threshold})"
            )
            
            all_chunks.extend(web_chunks)
        
        # Step 4: Sort by relevance (highest first)
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(
            f"Total chunks retrieved: {len(all_chunks)} "
            f"(corpus + web, sorted by relevance)"
        )
        
        return all_chunks
    
    def retrieve_formatted(
        self,
        query: str,
        budget: int
    ) -> str:
        """
        Retrieve and format results within STRICT token budget.
        
        This is the main method used by context assembly.
        CRITICAL: Must never exceed budget (550 tokens)
        
        Args:
            query: Search query
            budget: Token budget for retrieval section (550)
            
        Returns:
            Formatted retrieval string within budget
        """
        from src.core.prioritizer import get_prioritizer
        
        self.logger.info(f"Retrieving with STRICT budget: {budget} tokens")
        
        # Retrieve chunks (corpus + web)
        chunks = self.retrieve(query)
        
        if not chunks:
            self.logger.warning("No relevant documents found")
            return ""
        
        # Log chunk sources for transparency
        corpus_count = sum(1 for c in chunks if c.source.startswith('corpus:'))
        web_count = sum(1 for c in chunks if c.source.startswith('web:'))
        
        self.logger.info(f"Chunk breakdown: {corpus_count} corpus + {web_count} web")
        
        # Truncate to budget using prioritizer (keeps highest scoring)
        prioritizer = get_prioritizer()
        formatted = prioritizer.truncate_retrieval(chunks, budget)
        
        # CRITICAL VERIFICATION: Ensure we're within budget
        final_tokens = self.token_counter.count_tokens(formatted)
        
        if final_tokens > budget:
            self.logger.error(
                f"BUDGET VIOLATION: Retrieval section has {final_tokens} tokens "
                f"but budget is {budget}. Applying emergency truncation."
            )
            # Emergency truncation
            formatted = self.token_counter.truncate_to_budget(formatted, budget)
            final_tokens = self.token_counter.count_tokens(formatted)
        
        self.logger.info(
            f"Formatted retrieval: {final_tokens}/{budget} tokens "
            f"({final_tokens/budget*100:.1f}% of budget)"
        )
        
        return formatted
    
    def get_stats(self) -> Dict:
        """
        Get retrieval statistics.
        
        Returns:
            Dictionary with stats from all sources
        """
        return {
            'corpus_stats': self.vector_store.get_collection_stats(),
            'web_cache_stats': self.web_scraper.get_cache_stats()
        }


# Global singleton instance
_enhanced_retriever_instance: Optional[EnhancedRetriever] = None


def get_enhanced_retriever() -> EnhancedRetriever:
    """
    Get the global EnhancedRetriever instance.
    
    Returns:
        Singleton EnhancedRetriever instance
    """
    global _enhanced_retriever_instance
    if _enhanced_retriever_instance is None:
        _enhanced_retriever_instance = EnhancedRetriever()
    return _enhanced_retriever_instance


