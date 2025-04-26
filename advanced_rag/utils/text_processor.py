"""Text processing utilities."""
import logging
import re
from typing import Callable, Dict, List, Optional

from advanced_rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from advanced_rag.models.document import Document, DocumentMetadata

logger = logging.getLogger(__name__)


class TextProcessor:
    """Process and chunk text for the RAG system."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk a document into smaller pieces."""
        chunks = self._chunk_text(document.content)
        
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            metadata = DocumentMetadata(
                source_type=document.metadata.source_type,
                title=document.metadata.title,
                url=document.metadata.url,
                author=document.metadata.author,
                publish_date=document.metadata.publish_date,
                retrieved_date=document.metadata.retrieved_date,
                query=document.metadata.query,
                page_number=document.metadata.page_number,
                chunk_index=i,
            )
            
            chunked_doc = Document(
                content=chunk,
                metadata=metadata,
                embedding=None,  # Will be computed later
            )
            
            chunked_documents.append(chunked_doc)
            
        return chunked_documents
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to chunk at paragraph breaks first
        paragraphs = text.split("\n\n")
        
        # If paragraphs are still too long, split them further
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if para_size > self.chunk_size:
                # This paragraph is too long, so split it into sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    if sentence_size > self.chunk_size:
                        # Even the sentence is too long, split by chunk_size
                        for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                            sub_chunk = sentence[i:i + self.chunk_size]
                            chunks.append(sub_chunk)
                    elif current_size + sentence_size > self.chunk_size:
                        # Adding this sentence would exceed the chunk size, so start a new chunk
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = [sentence]
                        current_size = sentence_size
                    else:
                        # Add the sentence to the current chunk
                        current_chunk.append(sentence)
                        current_size += sentence_size
            elif current_size + para_size > self.chunk_size:
                # Adding this paragraph would exceed the chunk size, so start a new chunk
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                # Add the paragraph to the current chunk
                current_chunk.append(para)
                current_size += para_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and normalizing."""
        if not text:
            return ""
            
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_for_embeddings(self, text: str) -> str:
        """Normalize text for embedding generation."""
        # Clean the text first
        text = self.clean_text(text)
        
        # Remove markdown formatting
        text = re.sub(r'#+ ', '', text)  # Remove headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        
        # Limit the length for embeddings
        max_length = 8192  # Maximum context for most embedding models
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
