import logging
from typing import Dict, List, Optional
import os

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from advanced_rag.models.document import SourceType
from advanced_rag.rag.pipeline import RagPipeline
from advanced_rag.config import RERANKING_CONFIG

logger = logging.getLogger(__name__)

if RERANKING_CONFIG["enabled"]:
    try:
        from advanced_rag.rag.reranking_pipeline import ReRankingPipeline
    except ImportError:
        ReRankingPipeline = None
        logger.warning("ReRankingPipeline module not found, will use standard pipeline only")
else:
    ReRankingPipeline = None


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    use_web: bool = True
    refresh_cache: bool = False
    source_filter: Optional[str] = None
    use_reranking: bool = True


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    answer: str
    processing_time: float
    sources: List[Dict]


class RagAPI:
    
    def __init__(self, rag_pipeline: RagPipeline, reranking_pipeline: Optional["ReRankingPipeline"] = None):
        self.rag_pipeline = rag_pipeline
        self.reranking_pipeline = reranking_pipeline
        self.app = FastAPI(title="Advanced RAG API")
        self.templates = Jinja2Templates(directory="advanced_rag/ui/templates")
        
        # Set up routes
        self.setup_routes()
    
    def setup_routes(self):
        """Set up API routes."""
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            # Pass reranking config to template
            return self.templates.TemplateResponse("index.html", {
                "request": request,
                "reranking_enabled": RERANKING_CONFIG["enabled"] and self.reranking_pipeline is not None
            })
        
        @self.app.post("/api/query")
        async def query(query_request: QueryRequest):
            try:
                start_time = __import__('time').time()
                
                # Parse filter criteria
                filter_criteria = None
                if query_request.source_filter and query_request.source_filter != "all":
                    try:
                        source_type = SourceType(query_request.source_filter)
                        filter_criteria = {"source_type": source_type}
                    except ValueError:
                        pass
                
                # Choose pipeline based on request
                active_pipeline = self.reranking_pipeline if (
                    query_request.use_reranking and 
                    self.reranking_pipeline is not None and 
                    RERANKING_CONFIG["enabled"]
                ) else self.rag_pipeline
                
                # Process query
                answer, retrieved_docs = await active_pipeline.process_query(
                    query=query_request.query,
                    generate_answer=True,
                    use_web=query_request.use_web,
                    filter_criteria=filter_criteria,
                    refresh_cache=query_request.refresh_cache,
                )
                
                # Format sources for response
                sources = []
                for doc in retrieved_docs:
                    source = {
                        "title": doc.metadata.title,
                        "url": doc.metadata.url,
                        "source_type": doc.metadata.source_type.value 
                            if isinstance(doc.metadata.source_type, SourceType) else doc.metadata.source_type,
                        "relevance_score": doc.metadata.relevance_score,
                    }
                    sources.append(source)
                
                end_time = __import__('time').time()
                processing_time = end_time - start_time
                
                # Include pipeline type in the response
                return {
                    "query": query_request.query,
                    "answer": answer,
                    "processing_time": processing_time,
                    "sources": sources,
                    "pipeline_type": "reranking" if query_request.use_reranking and active_pipeline == self.reranking_pipeline else "standard"
                }
            
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/upload-documents")
        async def upload_documents(files: List[UploadFile] = File()):
            try:
                if not files or len(files) == 0:
                    raise HTTPException(status_code=400, detail="No files provided")
                
                processed_files = []
                
                for file in files:
                    content = await file.read()
                    file_info = {
                        "name": file.filename,
                        "content": content,
                        "content_type": file.content_type
                    }
                    processed_files.append(file_info)
                
                # Process documents with the standard pipeline
                documents = self.rag_pipeline.process_pdf_files(processed_files)
                
                # Set uploaded_pdfs flag on both pipelines
                self.rag_pipeline.uploaded_pdfs = True
                if self.reranking_pipeline:
                    self.reranking_pipeline.uploaded_pdfs = True
                
                return {
                    "status": "success",
                    "message": f"Successfully processed {len(files)} document(s)",
                    "num_chunks": len(documents)
                }
                
            except Exception as e:
                logger.error(f"Error uploading documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/clear-documents")
        async def clear_documents():
            try:
                await self.rag_pipeline.clear_documents()
                if self.reranking_pipeline:
                    await self.reranking_pipeline.clear_documents()
                
                return {"status": "success", "message": "Documents cleared successfully"}
                
            except Exception as e:
                logger.error(f"Error clearing documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def get_app(self):
        return self.app
