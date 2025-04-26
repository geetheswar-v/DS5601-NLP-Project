"""ReRanking RAG pipeline implementation."""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from advanced_rag.config import RERANKING_CONFIG
from advanced_rag.database.vector_store import VectorStore
from advanced_rag.llm.ollama_client import OllamaClient
from advanced_rag.models.document import Document, SourceType
from advanced_rag.rag.pipeline import RagPipeline
from advanced_rag.retrieval.reranker import DocumentReranker
from advanced_rag.retrieval.web_retrieval import WebRetriever
from advanced_rag.utils.cache_manager import CacheManager
from advanced_rag.utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class ReRankingPipeline(RagPipeline):
    """ReRanking RAG pipeline that implements two-stage retrieval."""
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        web_retriever: WebRetriever,
        vector_store: VectorStore,
        text_processor: TextProcessor,
        cache_manager: Optional[CacheManager] = None,
        reranker_model: str = RERANKING_CONFIG["model_name"],
        initial_retrieval_k: int = RERANKING_CONFIG["initial_retrieval_k"],
        final_top_k: int = RERANKING_CONFIG["final_top_k"],
    ):
        """
        Initialize the ReRanking pipeline.
        
        Args:
            ollama_client: Client for LLM interactions
            web_retriever: Retriever for web search
            vector_store: Vector database for document storage
            text_processor: Processor for text manipulation
            cache_manager: Optional cache manager
            reranker_model: Model name for reranking
            initial_retrieval_k: Number of documents to initially retrieve
            final_top_k: Number of documents to return after reranking
        """
        super().__init__(
            ollama_client=ollama_client,
            web_retriever=web_retriever,
            vector_store=vector_store,
            text_processor=text_processor,
            cache_manager=cache_manager,
        )
        
        self.reranker = DocumentReranker(model_name=reranker_model)
        self.initial_retrieval_k = initial_retrieval_k
        self.final_top_k = final_top_k
        logger.info(f"Initialized ReRanking pipeline with {reranker_model}")
    
    async def process_query(
        self, 
        query: str,
        generate_answer: bool = True,
        use_web: bool = True,
        filter_criteria: Optional[Dict] = None,
        refresh_cache: bool = False,
    ) -> Tuple[str, List[Document]]:
        """
        Process a query using two-stage retrieval with reranking.
        
        Args:
            query: The user query
            generate_answer: Whether to generate an answer or just return relevant documents
            use_web: Whether to retrieve new information from the web
            filter_criteria: Optional filter for vector search
            refresh_cache: Whether to force refresh the cache
            
        Returns:
            Tuple of (answer, retrieved_documents)
        """
        start_time = time.time()
        logger.info(f"Processing query with reranking: {query}")
        
        # Check cache first
        cache_key = f"reranking_answer:{query}"
        if not refresh_cache and self.cache_manager:
            cached = self.cache_manager.get(cache_key)
            if cached:
                logger.info(f"Using cached reranking answer for query: {query}")
                return cached
        
        # Retrieve a larger initial set of documents for reranking
        vector_docs = await self.vector_store.query(
            query, 
            top_k=self.initial_retrieval_k,
            filter_criteria=filter_criteria
        )
        logger.info(f"Retrieved {len(vector_docs)} initial documents from vector store for reranking")
        
        # Retrieve from web if needed and no PDFs uploaded
        if use_web and not self.uploaded_pdfs:
            web_docs = await self.web_retriever.retrieve(query)
            logger.info(f"Retrieved {len(web_docs)} documents from web")
            
            # Process and add web documents to vector store
            chunked_docs = []
            for doc in web_docs:
                chunked_docs.extend(self.text_processor.chunk_document(doc))
            
            await self.vector_store.add_documents(chunked_docs)
            
            # Get a new expanded set of documents for reranking
            all_docs = await self.vector_store.query(
                query, 
                top_k=self.initial_retrieval_k,
                filter_criteria=filter_criteria
            )
            logger.info(f"Retrieved {len(all_docs)} total documents for reranking")
        else:
            all_docs = vector_docs
        
        if not all_docs:
            logger.warning(f"No relevant documents found for query: {query}")
            answer = "I couldn't find any relevant information to answer your question." if generate_answer else ""
            return answer, []
        
        # Perform reranking
        reranked_docs = self.reranker.rerank(query, all_docs)
        logger.info(f"Reranked {len(reranked_docs)} documents")
        
        # Limit to final top_k
        top_docs = reranked_docs[:self.final_top_k]
        
        if generate_answer:
            contexts = [doc.content for doc in top_docs]
            
            system_prompt = """You are a helpful AI assistant for engineering students. 
Your answers should be:
1. Accurate and based on the provided context
2. Well-structured and clear
3. Concise but comprehensive
4. Include relevant technical details when appropriate
5. Cite sources when possible (using [n] notation if available in the context)

If information in the context is incomplete or contradictory, acknowledge this in your answer.
If the question cannot be answered based on the provided context, say so clearly.
"""
            answer = await self.ollama_client.agenerate_with_context(
                question=query,
                context=contexts,
                system=system_prompt,
            )
            
            if self.cache_manager:
                self.cache_manager.set(cache_key, (answer, top_docs))
            
            logger.info(f"Query processing with reranking completed in {time.time() - start_time:.2f}s")
            return answer, top_docs
        else:
            logger.info(f"Query processing with reranking completed in {time.time() - start_time:.2f}s")
            return "", top_docs
