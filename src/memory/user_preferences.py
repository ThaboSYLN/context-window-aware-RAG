"""
User Preferences Manager

Stores user-specific preferences and context that should persist
across exchanges but is separate from conversation history.

This version persists preferences to disk.
"""

import logging
import json
from typing import Dict, Optional, Any
from pathlib import Path


class UserPreferences:
    """
    Manages user preferences and context.
    
    This is lightweight context about the user that helps
    personalize responses without cluttering conversation memory.
    Persists to disk to maintain preferences across sessions.
    """
    
    def __init__(self, persist_file: str = "./data/memory/preferences.json"):
        """
        Initialize user preferences.
        
        Args:
            persist_file: Path to persistence file
        """
        self.logger = logging.getLogger(__name__)
        
        self.persist_file = Path(persist_file)
        self.preferences: Dict[str, Any] = {}
        
        # Ensure directory exists
        self.persist_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing preferences
        self._load_from_disk()
        
        self.logger.info(
            f"Initialized UserPreferences (loaded {len(self.preferences)} preferences)"
        )
    
    def _load_from_disk(self) -> None:
        """Load preferences from disk"""
        if self.persist_file.exists():
            try:
                with open(self.persist_file, 'r', encoding='utf-8') as f:
                    self.preferences = json.load(f)
                
                self.logger.debug(f"Loaded {len(self.preferences)} preferences from disk")
                
            except Exception as e:
                self.logger.error(f"Failed to load preferences from disk: {e}")
    
    def _save_to_disk(self) -> None:
        """Save preferences to disk"""
        try:
            with open(self.persist_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved {len(self.preferences)} preferences to disk")
            
        except Exception as e:
            self.logger.error(f"Failed to save preferences to disk: {e}")
    
    def set_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference.
        
        Args:
            key: Preference key
            value: Preference value
        """
        self.preferences[key] = value
        self._save_to_disk()
        self.logger.debug(f"Set preference: {key} = {value}")
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if key not found
            
        Returns:
            Preference value or default
        """
        return self.preferences.get(key, default)
    
    def has_preference(self, key: str) -> bool:
        """
        Check if a preference exists.
        
        Args:
            key: Preference key
            
        Returns:
            True if preference exists
        """
        return key in self.preferences
    
    def remove_preference(self, key: str) -> None:
        """
        Remove a preference.
        
        Args:
            key: Preference key
        """
        if key in self.preferences:
            del self.preferences[key]
            self._save_to_disk()
            self.logger.debug(f"Removed preference: {key}")
    
    def clear_all(self) -> None:
        """Clear all preferences"""
        self.preferences.clear()
        
        # Clear disk file
        if self.persist_file.exists():
            self.persist_file.unlink()
        
        self.logger.info("Cleared all preferences")
    
    def format_for_context(self) -> str:
        """
        Format preferences as context string.
        
        Returns:
            Formatted preferences
        """
        if not self.preferences:
            return ""
        
        parts = []
        for key, value in self.preferences.items():
            parts.append(f"{key}: {value}")
        
        return "User Context: " + ", ".join(parts)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all preferences.
        
        Returns:
            Dictionary of all preferences
        """
        return self.preferences.copy()


_user_preferences_instance: Optional[UserPreferences] = None


def get_user_preferences() -> UserPreferences:
    """
    Get the global UserPreferences instance.
    
    Returns:
        Singleton UserPreferences instance
    """
    global _user_preferences_instance
    if _user_preferences_instance is None:
        _user_preferences_instance = UserPreferences()
    return _user_preferences_instance

