"""
Web Scraper with Smart Caching

Scrapes web content based on user queries with intelligent caching:
- Automatically determines if web search is needed
- Caches results to avoid re-scraping identical queries
- Extracts clean text from HTML
- Respects rate limits and handles errors gracefully

Cache Strategy:
- Query-based caching (hash of normalized query)
- TTL (Time To Live) of 24 hours for cached results
- Automatic cache cleanup for expired entries
"""

import os
import hashlib
import logging
import json
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, quote_plus
from dotenv import load_dotenv

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False


class WebScraperCache:
    """
    Manages caching of web scraping results.
    
    Prevents re-scraping the same queries and provides TTL-based expiration.
    """
    
    def __init__(self, cache_dir: str = "./data/web_cache", ttl_hours: int = 24):
        """
        Initialize web scraper cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time to live for cached entries (hours)
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ttl_seconds = ttl_hours * 3600
        self.cache_index_file = self.cache_dir / "cache_index.json"
        
        # Load existing cache index
        self.cache_index = self._load_cache_index()
        
        # Cleanup expired entries
        self._cleanup_expired()
        
        self.logger.info(f"Initialized WebScraperCache (TTL: {ttl_hours}h, entries: {len(self.cache_index)})")
    
    def _load_cache_index(self) -> Dict:
        """Load cache index from disk"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load cache index: {e}")
                return {}
        return {}
    
    def _save_cache_index(self) -> None:
        """Save cache index to disk"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for query_hash, entry in self.cache_index.items():
            if current_time - entry['timestamp'] > self.ttl_seconds:
                expired_keys.append(query_hash)
                
                # Delete cache file
                cache_file = self.cache_dir / f"{query_hash}.json"
                if cache_file.exists():
                    cache_file.unlink()
        
        for key in expired_keys:
            del self.cache_index[key]
        
        if expired_keys:
            self._save_cache_index()
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent caching.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query
        """
        # Lowercase, strip whitespace, remove extra spaces
        normalized = ' '.join(query.lower().strip().split())
        return normalized
    
    def _hash_query(self, query: str) -> str:
        """
        Generate hash for query.
        
        Args:
            query: Normalized query
            
        Returns:
            Query hash (hex string)
        """
        return hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]
    
    def get(self, query: str) -> Optional[List[Dict]]:
        """
        Get cached results for query.
        
        Args:
            query: User query
            
        Returns:
            List of cached results or None if not found/expired
        """
        normalized = self._normalize_query(query)
        query_hash = self._hash_query(normalized)
        
        if query_hash not in self.cache_index:
            self.logger.debug(f"Cache MISS for query: {query}")
            return None
        
        entry = self.cache_index[query_hash]
        
        # Check if expired
        if time.time() - entry['timestamp'] > self.ttl_seconds:
            self.logger.debug(f"Cache EXPIRED for query: {query}")
            del self.cache_index[query_hash]
            self._save_cache_index()
            return None
        
        # Load cached results
        cache_file = self.cache_dir / f"{query_hash}.json"
        
        if not cache_file.exists():
            self.logger.warning(f"Cache file missing for query: {query}")
            del self.cache_index[query_hash]
            self._save_cache_index()
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            self.logger.info(f"Cache HIT for query: {query} ({len(results)} results)")
            return results
        except Exception as e:
            self.logger.error(f"Failed to load cache file: {e}")
            return None
    
    def set(self, query: str, results: List[Dict]) -> None:
        """
        Cache results for query.
        
        Args:
            query: User query
            results: Scraping results to cache
        """
        normalized = self._normalize_query(query)
        query_hash = self._hash_query(normalized)
        
        # Save results to file
        cache_file = self.cache_dir / f"{query_hash}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Update index
            self.cache_index[query_hash] = {
                'query': normalized,
                'timestamp': time.time(),
                'result_count': len(results)
            }
            
            self._save_cache_index()
            
            self.logger.info(f"Cached {len(results)} results for query: {query}")
        except Exception as e:
            self.logger.error(f"Failed to cache results: {e}")
    
    def clear(self) -> None:
        """Clear all cached results"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        
        self.cache_index.clear()
        self._save_cache_index()
        
        self.logger.info("Cleared all cache entries")
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'total_entries': len(self.cache_index),
            'cache_directory': str(self.cache_dir),
            'ttl_hours': self.ttl_seconds / 3600
        }


class WebScraper:
    """
    Web scraper with intelligent query-based scraping.
    
    Features:
    - Automatic Google search based on query
    - Clean text extraction from HTML
    - Smart caching to avoid re-scraping
    - Rate limiting and error handling
    """
    
    def __init__(
        self,
        max_results: int = 5,
        max_chars_per_page: int = 5000,
        timeout: int = 10
    ):
        """
        Initialize web scraper.
        
        Args:
            max_results: Maximum number of URLs to scrape per query
            max_chars_per_page: Maximum characters to extract per page
            timeout: Request timeout in seconds
        """
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        
        if not SCRAPING_AVAILABLE:
            self.logger.error("requests and beautifulsoup4 not available. Install with: pip install requests beautifulsoup4")
            raise ImportError("requests and beautifulsoup4 required for web scraping")
        
        self.max_results = max_results
        self.max_chars_per_page = max_chars_per_page
        self.timeout = timeout
        
        # Initialize cache
        self.cache = WebScraperCache()
        
        # User agent to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        self.logger.info(f"Initialized WebScraper (max_results={max_results}, max_chars={max_chars_per_page})")
    
    def _search_google(self, query: str) -> List[str]:
        """
        Search Google and extract result URLs.
        
        Args:
            query: Search query
            
        Returns:
            List of URLs
        """
        try:
            # Build Google search URL
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={self.max_results}"
            
            self.logger.debug(f"Searching Google: {query}")
            
            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract URLs from search results
            urls = []
            for link in soup.find_all('a'):
                href = link.get('href')
                
                if href and '/url?q=' in href:
                    # Extract actual URL from Google redirect
                    url = href.split('/url?q=')[1].split('&')[0]
                    
                    # Filter out Google-related URLs
                    if not any(domain in url for domain in ['google.com', 'youtube.com', 'facebook.com']):
                        urls.append(url)
                        
                        if len(urls) >= self.max_results:
                            break
            
            self.logger.info(f"Found {len(urls)} URLs from Google search")
            return urls
        
        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return []
    
    def _scrape_url(self, url: str) -> Optional[Dict]:
        """
        Scrape content from a single URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with scraped content or None if failed
        """
        try:
            self.logger.debug(f"Scraping URL: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = ' '.join(lines)
            
            # Truncate if too long
            if len(clean_text) > self.max_chars_per_page:
                clean_text = clean_text[:self.max_chars_per_page] + "..."
            
            # Get title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else urlparse(url).netloc
            
            result = {
                'url': url,
                'title': title_text,
                'content': clean_text,
                'scraped_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully scraped: {title_text} ({len(clean_text)} chars)")
            return result
        
        except Exception as e:
            self.logger.warning(f"Failed to scrape {url}: {e}")
            return None
    
    def scrape_for_query(self, query: str, use_cache: bool = True) -> List[Dict]:
        """
        Scrape web content for a query.
        
        This is the main method that:
        1. Checks cache first
        2. If not cached, searches Google
        3. Scrapes top results
        4. Caches results
        
        Args:
            query: User query
            use_cache: Whether to use cached results
            
        Returns:
            List of scraped content dictionaries
        """
        # Check cache first
        if use_cache:
            cached_results = self.cache.get(query)
            if cached_results is not None:
                return cached_results
        
        self.logger.info(f"Scraping web for query: {query}")
        
        # Search Google for URLs
        urls = self._search_google(query)
        
        if not urls:
            self.logger.warning("No URLs found from search")
            return []
        
        # Scrape each URL
        results = []
        for url in urls:
            scraped = self._scrape_url(url)
            if scraped:
                results.append(scraped)
            
            # Small delay to be polite
            time.sleep(0.5)
        
        # Cache results
        if results and use_cache:
            self.cache.set(query, results)
        
        self.logger.info(f"Scraped {len(results)} pages for query: {query}")
        return results
    
    def clear_cache(self) -> None:
        """Clear the scraping cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.get_stats()


# Global singleton instance
_web_scraper_instance: Optional[WebScraper] = None


def get_web_scraper() -> WebScraper:
    """
    Get the global WebScraper instance.
    
    Returns:
        Singleton WebScraper instance
    """
    global _web_scraper_instance
    if _web_scraper_instance is None:
        _web_scraper_instance = WebScraper()
    return _web_scraper_instance


