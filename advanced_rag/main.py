import logging
from pathlib import Path
import os
import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from advanced_rag.config import RERANKING_CONFIG, STATIC_DIRECTORY, LOGGING_CONFIG
from advanced_rag.database.vector_store import VectorStore
from advanced_rag.llm.ollama_client import OllamaClient
from advanced_rag.rag.pipeline import RagPipeline
from advanced_rag.retrieval.web_retrieval import WebRetriever
from advanced_rag.ui.api import RagAPI
from advanced_rag.utils.cache_manager import CacheManager
from advanced_rag.utils.logging_utils import setup_logging
from advanced_rag.utils.text_processor import TextProcessor

if RERANKING_CONFIG["enabled"]:
    try:
        from advanced_rag.rag.reranking_pipeline import ReRankingPipeline
    except ImportError:
        logging.getLogger(__name__).warning("ReRankingPipeline not available, will use standard pipeline only")
        RERANKING_CONFIG["enabled"] = False

logger = logging.getLogger(__name__)


def create_app(logging_enabled=True):
    setup_logging(enable=logging_enabled)
    logger.info("Initializing RAG components")
    cache_manager = CacheManager()
    ollama_client = OllamaClient()
    text_processor = TextProcessor()
    
    # Vector store
    vector_store = VectorStore(
        ollama_client=ollama_client,
        text_processor=text_processor,
    )
    
    # Web retriever
    web_retriever = WebRetriever(
        cache_manager=cache_manager,
        text_processor=text_processor,
    )
    
    # Standard RAG pipeline
    rag_pipeline = RagPipeline(
        ollama_client=ollama_client,
        web_retriever=web_retriever,
        vector_store=vector_store,
        text_processor=text_processor,
        cache_manager=cache_manager,
    )
    
    reranking_pipeline = None
    if RERANKING_CONFIG["enabled"]:
        try:
            logger.info("Initializing ReRanking pipeline")
            reranking_pipeline = ReRankingPipeline(
                ollama_client=ollama_client,
                web_retriever=web_retriever,
                vector_store=vector_store,
                text_processor=text_processor,
                cache_manager=cache_manager,
                reranker_model=RERANKING_CONFIG["model_name"],
                initial_retrieval_k=RERANKING_CONFIG["initial_retrieval_k"],
                final_top_k=RERANKING_CONFIG["final_top_k"],
            )
        except Exception as e:
            logger.error(f"Failed to initialize ReRanking pipeline: {e}")
            logger.warning("Continuing with standard RAG pipeline only")
    
    # Create API
    api = RagAPI(rag_pipeline=rag_pipeline, reranking_pipeline=reranking_pipeline)
    app = api.get_app()
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    if os.path.exists(STATIC_DIRECTORY):
        app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Run ContextAware APP (Advanced RAG)")
    parser.add_argument(
        "--logging", 
        action="store_true", 
        default=LOGGING_CONFIG["enabled"],
        help="Enable application logging"
    )
    parser.add_argument(
        "--no-logging", 
        action="store_false", 
        dest="logging",
        help="Disable application logging"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to"
    )
    
    args = parser.parse_args()
    app = create_app(logging_enabled=args.logging)
    
    log_level = "info" if args.logging else "error"
    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level)


if __name__ == "__main__":
    main()
