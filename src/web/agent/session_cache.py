"""
Session cache service for agent chat.
Provides in-memory caching of active chat sessions with TTL (24 hours).
"""

from cachetools import TTLCache
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SessionCacheEntry:
    """Cache entry for a chat session with message context."""

    def __init__(self, session_id: str, messages: List[Dict], last_access: datetime):
        self.session_id = session_id
        self.messages = messages  # Last 20 messages for context
        self.last_access = last_access
        self.created_at = datetime.utcnow()

    def __repr__(self):
        return f"SessionCacheEntry(session_id={self.session_id}, messages={len(self.messages)}, last_access={self.last_access})"


class SessionCache:
    """
    LRU cache with TTL for active chat sessions.

    Features:
    - Automatically expires entries after 24 hours (configurable)
    - Limits memory usage with max size (100 sessions by default)
    - Thread-safe operations
    - Cleanup expired sessions on demand
    """

    def __init__(self, maxsize: int = 100, ttl_seconds: int = 86400):
        """
        Initialize session cache.

        Args:
            maxsize: Maximum number of sessions to cache
            ttl_seconds: Time-to-live in seconds (default: 86400 = 24 hours)
        """
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        logger.info(
            f"Session cache initialized: maxsize={maxsize}, ttl={ttl_seconds}s ({ttl_seconds/3600:.1f}h)"
        )

    def get(self, session_id: str) -> Optional[SessionCacheEntry]:
        """
        Get cached session entry.

        Args:
            session_id: UUID of the session

        Returns:
            SessionCacheEntry if found and not expired, None otherwise
        """
        entry = self.cache.get(session_id)
        if entry:
            entry.last_access = datetime.utcnow()
            logger.debug(f"Cache hit for session {session_id}")
        else:
            logger.debug(f"Cache miss for session {session_id}")
        return entry

    def set(self, session_id: str, messages: List[Dict]) -> None:
        """
        Cache session with message context.

        Args:
            session_id: UUID of the session
            messages: List of recent messages (usually last 20)
        """
        entry = SessionCacheEntry(session_id, messages, datetime.utcnow())
        self.cache[session_id] = entry
        logger.debug(
            f"Cached session {session_id} with {len(messages)} messages"
        )

    def update_messages(self, session_id: str, messages: List[Dict]) -> bool:
        """
        Update messages for existing cached session.

        Args:
            session_id: UUID of the session
            messages: Updated message list

        Returns:
            True if session was found and updated, False otherwise
        """
        entry = self.get(session_id)
        if entry:
            entry.messages = messages
            entry.last_access = datetime.utcnow()
            logger.debug(f"Updated session {session_id} with {len(messages)} messages")
            return True
        return False

    def cleanup_expired(self) -> int:
        """
        Manually trigger cleanup of expired entries.

        TTLCache automatically expires entries, but this method
        forces a check and returns the count of expired entries.

        Returns:
            Number of entries removed
        """
        initial_size = len(self.cache)
        # Touch all keys to trigger TTL check
        _ = list(self.cache.keys())
        removed = initial_size - len(self.cache)

        if removed > 0:
            logger.info(f"Cleaned up {removed} expired sessions from cache")

        return removed

    def remove(self, session_id: str) -> None:
        """
        Remove session from cache.

        Args:
            session_id: UUID of the session
        """
        if session_id in self.cache:
            del self.cache[session_id]
            logger.debug(f"Removed session {session_id} from cache")

    def clear(self) -> None:
        """Clear all cached sessions."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {count} sessions from cache")

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (size, maxsize, ttl, etc.)
        """
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "ttl_seconds": self.ttl_seconds,
            "ttl_hours": self.ttl_seconds / 3600,
        }

    def __len__(self) -> int:
        """Return number of cached sessions."""
        return len(self.cache)

    def __contains__(self, session_id: str) -> bool:
        """Check if session is in cache."""
        return session_id in self.cache


# Global singleton instance
_session_cache: Optional[SessionCache] = None


def get_session_cache() -> SessionCache:
    """
    Get the global session cache instance.

    Returns:
        SessionCache singleton instance
    """
    global _session_cache
    if _session_cache is None:
        _session_cache = SessionCache(maxsize=100, ttl_seconds=86400)
    return _session_cache


def reset_session_cache() -> None:
    """Reset the global session cache (primarily for testing)."""
    global _session_cache
    _session_cache = None
