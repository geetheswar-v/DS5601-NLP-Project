import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from advanced_rag.config import CACHE_DIR, CACHE_EXPIRY
from advanced_rag.models.document import Document

logger = logging.getLogger(__name__)


class CacheManager:
    
    def __init__(self, cache_dir: Path = CACHE_DIR, expiry: int = CACHE_EXPIRY):
        self.cache_dir = cache_dir
        self.expiry = expiry
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Ensure the cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the path for a cache key."""
        # Create a valid filename from the key
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        return self.cache_dir / f"{safe_key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if it exists and is not expired."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            
            # Check if cache is expired
            if time.time() - data["timestamp"] > self.expiry:
                logger.debug(f"Cache expired for key: {key}")
                return None
            
            logger.debug(f"Cache hit for key: {key}")
            return data["value"]
        except Exception as e:
            logger.error(f"Error loading from cache for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with current timestamp."""
        cache_path = self._get_cache_path(key)
        
        try:
            data = {
                "timestamp": time.time(),
                "value": value,
            }
            
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
                
            logger.debug(f"Cached data for key: {key}")
        except Exception as e:
            logger.error(f"Error caching data for key {key}: {e}")
    
    def clear(self, key: Optional[str] = None) -> None:
        """Clear specific cache item or all cache if key is None."""
        if key:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                os.remove(cache_path)
                logger.debug(f"Cleared cache for key: {key}")
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                os.remove(cache_file)
            logger.debug("Cleared all cache")
    
    def clear_expired(self) -> None:
        """Clear all expired cache items."""
        now = time.time()
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                
                if now - data["timestamp"] > self.expiry:
                    os.remove(cache_file)
                    logger.debug(f"Removed expired cache: {cache_file.name}")
            except Exception as e:
                logger.error(f"Error checking cache expiry for {cache_file}: {e}")
