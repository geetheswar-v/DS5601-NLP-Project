import logging
from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

from advanced_rag.models.document import Document

logger = logging.getLogger(__name__)

class DocumentReranker:
    """We adeed a simple cross=encoder based reranker to the pipeline.
    In future we thought of adding a more complex reranker based on the
    Our Project Goal such that fine tune and etc"""
    
    def __init__(self, model_name: str):
        """
        Initialize the reranker with a cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        logger.info(f"Initializing document reranker with model: {model_name}")
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The user query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents with updated relevance scores
        """
        if not documents:
            return []
            
        # Prepare document-query pairs for scoring
        pairs = [(query, doc.content) for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Update document relevance scores
        scored_docs = []
        for i, doc in enumerate(documents):
            new_doc = Document(
                content=doc.content,
                metadata=doc.metadata,
                embedding=doc.embedding
            )
            new_doc.metadata.relevance_score = float(scores[i])
            scored_docs.append(new_doc)
        
        # Sort by relevance score in descending order
        reranked_docs = sorted(scored_docs, key=lambda x: x.metadata.relevance_score, reverse=True)
        
        logger.debug(f"Reranked {len(reranked_docs)} documents")
        return reranked_docs
