"""
Conversation Memory Manager

Manages short-term conversation state including:
- Recent user-assistant exchanges
- Conversation context
- Session information

This version persists memory to disk so it survives between CLI runs.
"""

import logging
import json
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque
from pathlib import Path


class ConversationMemory:
    """
    Manages conversation history and context.
    
    Uses a FIFO queue to maintain recent exchanges within token budget.
    Persists to disk to maintain context across CLI invocations.
    """
    
    def __init__(
        self,
        max_exchanges: int = 5,
        persist_file: str = "./data/memory/conversation.json"
    ):
        """
        Initialize conversation memory.
        
        Args:
            max_exchanges: Maximum number of exchanges to keep
            persist_file: Path to persistence file
        """
        self.logger = logging.getLogger(__name__)
        
        self.max_exchanges = max_exchanges
        self.persist_file = Path(persist_file)
        
        # Ensure directory exists
        self.persist_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory or start fresh
        self.exchanges = deque(maxlen=max_exchanges)
        self.session_start = datetime.now()
        self.total_exchanges = 0
        
        self._load_from_disk()
        
        self.logger.info(
            f"Initialized ConversationMemory "
            f"(max_exchanges={max_exchanges}, loaded={len(self.exchanges)} exchanges)"
        )
    
    def _load_from_disk(self) -> None:
        """Load conversation history from disk"""
        if self.persist_file.exists():
            try:
                with open(self.persist_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore exchanges
                for exchange in data.get('exchanges', []):
                    self.exchanges.append(exchange)
                
                self.total_exchanges = data.get('total_exchanges', 0)
                
                # Parse session start
                session_str = data.get('session_start')
                if session_str:
                    self.session_start = datetime.fromisoformat(session_str)
                
                self.logger.debug(f"Loaded {len(self.exchanges)} exchanges from disk")
                
            except Exception as e:
                self.logger.error(f"Failed to load memory from disk: {e}")
    
    def _save_to_disk(self) -> None:
        """Save conversation history to disk"""
        try:
            data = {
                'exchanges': list(self.exchanges),
                'total_exchanges': self.total_exchanges,
                'session_start': self.session_start.isoformat()
            }
            
            with open(self.persist_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved {len(self.exchanges)} exchanges to disk")
            
        except Exception as e:
            self.logger.error(f"Failed to save memory to disk: {e}")
    
    def add_exchange(self, user_message: str, assistant_response: str) -> None:
        """
        Add a user-assistant exchange to memory.
        
        Args:
            user_message: What the user said
            assistant_response: How the assistant responded
        """
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'assistant': assistant_response
        }
        
        self.exchanges.append(exchange)
        self.total_exchanges += 1
        
        # Persist to disk
        self._save_to_disk()
        
        self.logger.debug(f"Added exchange to memory (total in memory: {len(self.exchanges)})")
    
    def get_recent_exchanges(self, n: Optional[int] = None) -> List[Dict]:
        """
        Get recent exchanges.
        
        Args:
            n: Number of recent exchanges to get (None = all)
            
        Returns:
            List of exchange dictionaries
        """
        if n is None:
            return list(self.exchanges)
        
        return list(self.exchanges)[-n:] if n > 0 else []
    
    def format_for_context(self) -> str:
        """
        Format memory as a string for context assembly.
        
        Returns:
            Formatted conversation history
        """
        if not self.exchanges:
            return ""
        
        formatted_parts = []
        
        for i, exchange in enumerate(self.exchanges, 1):
            formatted_parts.append(
                f"Exchange {i}:\n"
                f"User: {exchange['user']}\n"
                f"Assistant: {exchange['assistant']}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def get_memory_items(self) -> List[str]:
        """
        Get memory as list of strings for prioritization.
        
        Returns:
            List of formatted exchanges (newest last)
        """
        items = []
        
        for exchange in self.exchanges:
            item = f"User: {exchange['user']}\nAssistant: {exchange['assistant']}"
            items.append(item)
        
        return items
    
    def clear(self) -> None:
        """Clear all conversation memory"""
        self.exchanges.clear()
        self.total_exchanges = 0
        self.session_start = datetime.now()
        
        # Clear disk file
        if self.persist_file.exists():
            self.persist_file.unlink()
        
        self.logger.info("Cleared conversation memory")
    
    def get_stats(self) -> Dict:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        return {
            'exchanges_in_memory': len(self.exchanges),
            'total_exchanges': self.total_exchanges,
            'session_duration': str(datetime.now() - self.session_start)
        }


_conversation_memory_instance: Optional[ConversationMemory] = None


def get_conversation_memory() -> ConversationMemory:
    """
    Get the global ConversationMemory instance.
    
    Returns:
        Singleton ConversationMemory instance
    """
    global _conversation_memory_instance
    if _conversation_memory_instance is None:
        _conversation_memory_instance = ConversationMemory()
    return _conversation_memory_instance

