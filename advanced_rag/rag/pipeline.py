import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from advanced_rag.database.vector_store import VectorStore
from advanced_rag.llm.ollama_client import OllamaClient
from advanced_rag.models.document import Document, SourceType
from advanced_rag.retrieval.web_retrieval import WebRetriever
from advanced_rag.retrieval.pdf_retrieval import DocumentRetriever
from advanced_rag.utils.cache_manager import CacheManager
from advanced_rag.utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class RagPipeline:
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        web_retriever: WebRetriever,
        vector_store: VectorStore,
        text_processor: TextProcessor,
        cache_manager: Optional[CacheManager] = None,
        document_retriever: Optional[DocumentRetriever] = None,
    ):
        self.ollama_client = ollama_client
        self.web_retriever = web_retriever
        self.vector_store = vector_store
        self.text_processor = text_processor
        self.cache_manager = cache_manager
        self.document_retriever = document_retriever or DocumentRetriever(
            cache_manager=cache_manager, 
            text_processor=text_processor
        )
        self.uploaded_pdfs = False
    
    def process_pdf_files(self, files: List[Dict], query: Optional[str] = None) -> List[Document]:
        """
        Process document files and add them to the vector store.
        
        Args:
            files: List of dictionaries with file info
            query: Optional query context
            
        Returns:
            List of processed documents
        """
        # Process the files
        documents = self.document_retriever.process_multiple_documents(files, query)
        
        # Chunk the documents for better retrieval
        chunked_docs = []
        for doc in documents:
            chunked_docs.extend(self.text_processor.chunk_document(doc))
        
        # Add to vector store
        if chunked_docs:
            # This needs to be run in an event loop, but we're calling this from sync code
            # So we create a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a task
                    asyncio.create_task(self.vector_store.add_documents(chunked_docs))
                else:
                    # We're in a sync context, run the coroutine
                    loop.run_until_complete(self.vector_store.add_documents(chunked_docs))
            except RuntimeError:
                # No event loop exists, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.vector_store.add_documents(chunked_docs))
            
            # Mark that we have documents uploaded
            self.uploaded_pdfs = True
        
        return chunked_docs
    
    async def process_query(
        self, 
        query: str,
        generate_answer: bool = True,
        use_web: bool = True,
        filter_criteria: Optional[Dict] = None,
        refresh_cache: bool = False,
    ) -> Tuple[str, List[Document]]:
        """
        Process a query using the RAG pipeline.
        
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
        logger.info(f"Processing query: {query}")
        
        # First, check if we've already answered this query
        cache_key = f"answer:{query}"
        if not refresh_cache and self.cache_manager:
            cached = self.cache_manager.get(cache_key)
            if cached:
                logger.info(f"Using cached answer for query: {query}")
                return cached
        
        # Retrieve documents from vector store
        vector_docs = await self.vector_store.query(query, filter_criteria=filter_criteria)
        logger.info(f"Retrieved {len(vector_docs)} documents from vector store")
        
        # Retrieve documents from web if requested and no PDFs are uploaded
        if use_web and not self.uploaded_pdfs:
            web_docs = await self.web_retriever.retrieve(query)
            logger.info(f"Retrieved {len(web_docs)} documents from web")
            
            # Process and add web documents to vector store for future use
            chunked_docs = []
            for doc in web_docs:
                chunked_docs.extend(self.text_processor.chunk_document(doc))
            
            await self.vector_store.add_documents(chunked_docs)
            
            # Get relevant documents again with the new additions
            all_docs = await self.vector_store.query(query, filter_criteria=filter_criteria)
            logger.info(f"Retrieved {len(all_docs)} documents from combined sources")
        else:
            all_docs = vector_docs
        
        if not all_docs:
            logger.warning(f"No relevant documents found for query: {query}")
            answer = "I couldn't find any relevant information to answer your question." if generate_answer else ""
            return answer, []
        
        # Sort documents by relevance score
        sorted_docs = sorted(
            all_docs, 
            key=lambda doc: doc.metadata.relevance_score or 0, 
            reverse=True
        )
        
        if generate_answer:
            contexts = [doc.content for doc in sorted_docs]
            
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
                self.cache_manager.set(cache_key, (answer, sorted_docs))
            
            logger.info(f"Query processing completed in {time.time() - start_time:.2f}s")
            return answer, sorted_docs
        else:
            logger.info(f"Query processing completed in {time.time() - start_time:.2f}s")
            return "", sorted_docs
    
    async def refine_query(self, query: str) -> str:
        system_prompt = """You are a search query optimization assistant. 
Your task is to refine and expand the user's query to improve search results.
Follow these guidelines:

1. Add relevant domain-specific terminology
2. Include important synonyms or related terms
3. Clarify ambiguous terms
4. Break down complex queries into clearer language
5. Keep the refined query concise (max 2-3 sentences)
6. Only output the refined query, nothing else

Example:
Original: "quantum computing basics"
Refined: "quantum computing fundamentals qubits superposition quantum gates entanglement"
"""
        
        prompt = f"Original query: {query}\n\nRefined query:"
        
        refined_query = await self.ollama_client.agenerate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.3,
        )
        
        if not refined_query or len(refined_query) < len(query) / 2:
            return query
            
        logger.debug(f"Refined query: '{query}' -> '{refined_query}'")
        return refined_query
    
    async def generate_search_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple search queries from the original query to improve coverage.
        
        Args:
            query: The original user query
            num_queries: Number of queries to generate
            
        Returns:
            List of search queries
        """
        system_prompt = f"""You are a search query generation assistant.
Your task is to create {num_queries} different search queries based on the user's original query.
Each query should:

1. Explore a different aspect or perspective of the original query
2. Use different terminology or phrasing
3. Be concise and focused
4. Target technical/engineering content when appropriate

Format your response as a numbered list, with one query per line.
Only output the numbered queries, nothing else.
"""
        
        prompt = f"Original query: {query}\n\nGenerate {num_queries} different search queries:"
        
        generated_text = await self.ollama_client.agenerate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.7,  # Allow some creativity
        )
        
        # Parse the numbered list
        queries = []
        for line in generated_text.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, num_queries + 1)):
                query_text = line[line.find('.') + 1:].strip()
                if query_text:
                    queries.append(query_text)
        
        # Add the original query if we don't have enough
        if not queries:
            return [query]
            
        # Ensure we don't exceed the requested number
        queries = queries[:num_queries]
        
        logger.debug(f"Generated {len(queries)} alternative search queries")
        return queries

    async def clear_documents(self, filter_criteria: Optional[Dict] = None):
        """
        Clear documents from the vector store and reset the uploaded_pdfs flag.
        
        Args:
            filter_criteria: Optional filter criteria to clear only specific documents
        """
        await self.vector_store.clear_documents(filter_criteria)
        if filter_criteria is None or filter_criteria.get("source_type") == SourceType.PDF:
            self.uploaded_pdfs = False
        
        logger.info(f"Cleared documents with filter: {filter_criteria}")
        return True
