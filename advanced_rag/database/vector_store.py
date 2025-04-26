"""Vector database for storing and retrieving document embeddings."""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import retry, stop_after_attempt, wait_exponential

from advanced_rag.config import TOP_K_RETRIEVAL, VECTOR_DB_PATH
from advanced_rag.llm.ollama_client import OllamaClient
from advanced_rag.models.document import Document, DocumentMetadata, SourceType
from advanced_rag.utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database for storing and retrieving document embeddings."""
    
    def __init__(
        self, 
        ollama_client: OllamaClient,
        text_processor: TextProcessor,
        collection_name: str = "advanced_rag",
        path: str = str(VECTOR_DB_PATH),
    ):
        self.ollama_client = ollama_client
        self.text_processor = text_processor
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.client = QdrantClient(path=path)
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Ensure the vector collection exists."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Get sample embedding to determine vector size
                sample_embedding = self.ollama_client.get_embeddings("sample text")[0]
                vector_size = len(sample_embedding)
                
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created vector collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring vector collection: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            return []
        
        points = []
        for i, doc in enumerate(documents):
            # Get or generate embedding
            if doc.embedding is None:
                normalized_text = self.text_processor.normalize_for_embeddings(doc.content)
                embeddings = await self.ollama_client.aget_embeddings(normalized_text)
                doc.embedding = embeddings[0]
            
            # Create point ID
            point_id = len(points) + 1
            
            # Create point
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=doc.embedding,
                    payload={
                        "content": doc.content,
                        "metadata": doc.metadata.to_dict(),
                    },
                )
            )
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return [str(p.id) for p in points]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def query(
        self, 
        query: str, 
        top_k: int = TOP_K_RETRIEVAL,
        filter_criteria: Optional[Dict] = None,
    ) -> List[Document]:
        """Query the vector store for similar documents."""
        # Get query embedding
        embeddings = await self.ollama_client.aget_embeddings(query)
        query_embedding = embeddings[0]
        
        # Build filter if provided
        search_filter = None
        if filter_criteria:
            filter_conditions = []
            
            for field, value in filter_criteria.items():
                if field == "source_type" and isinstance(value, (str, SourceType)):
                    source_value = value.value if isinstance(value, SourceType) else value
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{field}",
                            match=models.MatchValue(value=source_value),
                        )
                    )
                elif field == "date_range" and isinstance(value, dict):
                    start = value.get("start")
                    end = value.get("end")
                    if start and end:
                        filter_conditions.append(
                            models.FieldCondition(
                                key="metadata.publish_date",
                                range=models.Range(
                                    gte=start,
                                    lte=end,
                                ),
                            )
                        )
            
            if filter_conditions:
                search_filter = models.Filter(
                    must=filter_conditions,
                )
        
        # Query the vector store
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=search_filter,
        )
        
        # Convert results to Document objects
        documents = []
        for result in search_results:
            content = result.payload.get("content", "")
            metadata_dict = result.payload.get("metadata", {})
            metadata = DocumentMetadata.from_dict(metadata_dict)
            
            # Add relevance score to metadata
            metadata.relevance_score = result.score
            
            doc = Document(
                content=content,
                metadata=metadata,
                embedding=None,  # No need to store the embedding again
            )
            documents.append(doc)
        
        return documents
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def clear_documents(self, filter_criteria: Optional[Dict] = None):
        """
        Clear documents from the vector store.
        
        Args:
            filter_criteria: Optional filter to clear only specific documents
        """
        try:
            if filter_criteria:
                # Convert filter criteria to Qdrant filter format
                qdrant_filter = self._build_filter(filter_criteria)
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=qdrant_filter)
                )
            else:
                # Recreate the collection to clear all data
                self.client.delete_collection(collection_name=self.collection_name)
                self._ensure_collection()
            
            logger.info(f"Cleared documents from vector store with filter: {filter_criteria}")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            raise
    
    def clear(self) -> None:
        """Clear the vector store."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            logger.info(f"Cleared vector collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing vector collection: {e}")
            raise
