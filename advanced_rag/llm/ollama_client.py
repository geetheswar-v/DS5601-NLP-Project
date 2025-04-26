import json
import logging
from typing import Dict, List, Optional, Union

import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from advanced_rag.config import OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class OllamaClient:
    """This has both async and sync methods for generation of text and embeddings."""
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.generate_endpoint = f"{base_url}/api/generate"
        self.embeddings_endpoint = f"{base_url}/api/embeddings"
        self._check_models_available()
    
    def _check_models_available(self) -> None:
        """Check if required models are available in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            missing_models = []
            if LLM_MODEL not in available_models:
                missing_models.append(LLM_MODEL)
            if EMBEDDING_MODEL not in available_models:
                missing_models.append(EMBEDDING_MODEL)
                
            if missing_models:
                logger.warning(f"Missing required models: {', '.join(missing_models)}")
                logger.info(f"Please pull missing models using: ollama pull {' '.join(missing_models)}")
        except Exception as e:
            logger.error(f"Failed to check available models: {e}")
            logger.warning("Is Ollama running? Start it with 'ollama serve'")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(
        self,
        prompt: str,
        model: str = LLM_MODEL,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text using Ollama LLM."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        if context:
            payload["context"] = context
            
        try:
            response = requests.post(self.generate_endpoint, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_with_context(
        self,
        question: str,
        context: List[str],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text with context for RAG."""
        combined_context = "\n\n".join(context)
        prompt = f"""Please answer based on the following context (Note: the answer must be based on the context):

Context:
{combined_context}

Question: {question}

Answer:"""

        return self.generate(
            prompt=prompt,
            model=LLM_MODEL,
            system=system or "You are a helpful, accurate, and concise AI assistant for engineering students.",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_embeddings(self, texts: Union[str, List[str]], model: str = EMBEDDING_MODEL) -> List[List[float]]:
        """Get embeddings for text or list of texts."""
        if isinstance(texts, str):
            texts = [texts]
            
        results = []
        for text in texts:
            payload = {
                "model": model,
                "prompt": text,
            }
            
            try:
                response = requests.post(self.embeddings_endpoint, json=payload)
                response.raise_for_status()
                embedding = response.json().get("embedding", [])
                results.append(embedding)
            except Exception as e:
                logger.error(f"Error getting embeddings: {e}")
                raise
                
        return results
    
    async def agenerate(
        self,
        prompt: str,
        model: str = LLM_MODEL,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text asynchronously using Ollama LLM."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        if context:
            payload["context"] = context
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.generate_endpoint, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error {response.status}: {error_text}")
                    result = await response.json()
                    return result.get("response", "")
        except Exception as e:
            logger.error(f"Error generating text asynchronously: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def agenerate_with_context(
        self,
        question: str,
        context: List[str],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text asynchronously with context for RAG."""
        combined_context = "\n\n".join(context)
        prompt = f"""Please answer based on the following context:

Context:
{combined_context}

Question: {question}

Answer:"""

        return await self.agenerate(
            prompt=prompt,
            model=LLM_MODEL,
            system=system or "You are a helpful, accurate, and concise AI assistant for engineering students.",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    async def aget_embeddings(self, texts: Union[str, List[str]], model: str = EMBEDDING_MODEL) -> List[List[float]]:
        """Get embeddings asynchronously for text or list of texts."""
        if isinstance(texts, str):
            texts = [texts]
            
        results = []
        async with aiohttp.ClientSession() as session:
            for text in texts:
                payload = {
                    "model": model,
                    "prompt": text,
                }
                
                try:
                    async with session.post(self.embeddings_endpoint, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Error {response.status}: {error_text}")
                        result = await response.json()
                        embedding = result.get("embedding", [])
                        results.append(embedding)
                except Exception as e:
                    logger.error(f"Error getting embeddings asynchronously: {e}")
                    raise
                    
        return results
