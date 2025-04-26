import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import arxiv
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib.parse import quote_plus

from advanced_rag.config import (
    CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    SOURCES,
    USER_AGENT,
)
from advanced_rag.models.document import Document, DocumentMetadata, SourceType

logger = logging.getLogger(__name__)

class WebRetriever:
    """Component for retrieving content from various web sources."""
    
    def __init__(self, cache_manager=None, text_processor=None):
        self.cache_manager = cache_manager
        self.text_processor = text_processor
        self.sources = SOURCES
        self.headers = {
            "User-Agent": USER_AGENT,
        }
    
    async def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents from configured web sources.
        
        Args:
            query: The search query
            
        Returns:
            List of retrieved documents
        """
        tasks = []
        if self.sources["wikipedia"]["enabled"]:
            tasks.append(self.search_wikipedia(query))
        
        if self.sources["arxiv"]["enabled"]:
            tasks.append(self.search_arxiv(query))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        documents = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error retrieving results: {result}")
                continue
            documents.extend(result)
            
        return documents
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def search_wikipedia(self, query: str) -> List[Document]:
        """Search Wikipedia for the query."""
        if self.cache_manager:
            cached = self.cache_manager.get(f"wiki:{query}")
            if cached:
                return cached
        
        max_results = self.sources["wikipedia"]["max_results"]
        language = self.sources["wikipedia"]["language"]
        
        search_url = f"https://{language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, headers=self.headers, timeout=REQUEST_TIMEOUT) as response:
                    if response.status != 200:
                        logger.error(f"Wikipedia search failed with status {response.status}")
                        return []
                    
                    search_data = await response.json()
                    search_results = search_data.get("query", {}).get("search", [])
                    
                    if not search_results:
                        return []
                    
                    # Get content for each page in parallel
                    tasks = []
                    for result in search_results:
                        page_id = result["pageid"]
                        tasks.append(self._get_wikipedia_content(session, language, page_id, result["title"]))
                        
                    contents = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    documents = []
                    for i, content in enumerate(contents):
                        if isinstance(content, Exception):
                            logger.error(f"Error retrieving Wikipedia content: {content}")
                            continue
                            
                        if content:
                            result = search_results[i]
                            doc = Document(
                                content=content,
                                metadata=DocumentMetadata(
                                    source_type=SourceType.WIKIPEDIA,
                                    title=result["title"],
                                    url=f"https://{language}.wikipedia.org/?curid={result['pageid']}",
                                    author="Wikipedia Contributors",
                                    publish_date=None,
                                    retrieved_date=time.time(),
                                    query=query,
                                )
                            )
                            documents.append(doc)
                            
                    if self.cache_manager:
                        self.cache_manager.set(f"wiki:{query}", documents)
                        
                    return documents
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            raise
    
    async def _get_wikipedia_content(self, session: aiohttp.ClientSession, language: str, page_id: int, title: str) -> str:
        """Get content of a Wikipedia page."""
        url = f"https://{language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "pageids": page_id,
        }
        
        try:
            async with session.get(url, params=params, headers=self.headers, timeout=REQUEST_TIMEOUT) as response:
                if response.status != 200:
                    logger.error(f"Wikipedia content retrieval failed with status {response.status}")
                    return ""
                
                data = await response.json()
                page_data = data.get("query", {}).get("pages", {}).get(str(page_id), {})
                extract = page_data.get("extract", "")
                
                if not extract:
                    return ""
                
                # Add title as header
                content = f"# {title}\n\n{extract}"
                return content
        except Exception as e:
            logger.error(f"Error getting Wikipedia content: {e}")
            raise
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def search_arxiv(self, query: str) -> List[Document]:
        """Search arXiv for the query."""
        if self.cache_manager:
            cached = self.cache_manager.get(f"arxiv:{query}")
            if cached:
                return cached
        
        max_results = self.sources["arxiv"]["max_results"]
        categories = self.sources["arxiv"]["categories"]
        
        try:
            # This uses a synchronous client but it's relatively fast
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )
            
            results = list(search.results())
            
            documents = []
            for result in results:
                # Check if the paper is in relevant categories
                if categories and not any(cat in result.categories for cat in categories):
                    continue
                
                summary = result.summary.replace("\n", " ").strip()
                
                content = (
                    f"# {result.title}\n\n"
                    f"**Authors:** {', '.join(author.name for author in result.authors)}\n\n"
                    f"**Categories:** {', '.join(result.categories)}\n\n"
                    f"**Published:** {result.published.strftime('%Y-%m-%d')}\n\n"
                    f"**Abstract:**\n{summary}"
                )
                
                doc = Document(
                    content=content,
                    metadata=DocumentMetadata(
                        source_type=SourceType.ARXIV,
                        title=result.title,
                        url=result.entry_id,
                        author=", ".join(author.name for author in result.authors),
                        publish_date=result.published.timestamp(),
                        retrieved_date=time.time(),
                        query=query,
                    )
                )
                documents.append(doc)
                
            if self.cache_manager:
                self.cache_manager.set(f"arxiv:{query}", documents)
                
            return documents
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            raise
