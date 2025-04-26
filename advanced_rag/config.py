from pathlib import Path
from typing import Dict, List, Optional
import os

# Base directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"
VECTOR_DB_PATH = DATA_DIR / "vector_db"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIRECTORY = os.path.join(BASE_DIR, "static")

# Logging configuration
LOGGING_CONFIG = {
    "enabled": True,
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None,
}

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, VECTOR_DB_PATH]:
    directory.mkdir(parents=True, exist_ok=True)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG configuration
LLM_MODEL = "granite3.3:2b"
EMBEDDING_MODEL = "granite-embedding:30m"

# RAGAS configuration
RAGAS_LLM_MODEL = "gemma3" # 8b model
RAGAS_EMBEDDING_MODEL = ""

# RAG DOC configuration
MAX_DOCUMENTS = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# Re-ranking configuration (NEW)
RERANKING_CONFIG = {
    "enabled": False,
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "initial_retrieval_k": 20,
    "final_top_k": TOP_K_RETRIEVAL,
}

# Web retrieval configuration
CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 10
USER_AGENT = "AdvancedRAGBot/1.0 (Educational Research Tool)"

# Sources configuration
SOURCES = {
    "arxiv": {
        "enabled": True,
        "max_results": 5,
        "categories": [
            "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO",
            "eess.AS", "eess.IV", "eess.SP", "eess.SY",
            "math.OC", "stat.ML"
        ]
    },
    "wikipedia": {
        "enabled": True,
        "max_results": 3,
        "language": "en"
    }
    # removed sources when uploading it to github
}

# CLI configuration
CLI_CONFIG = {
    "use_web": True,
    "web_source": "wikipedia",
    "history_size": 10,
}

# Evaluation configuration
EVALUATION = {
    "ragas": {
        "enabled": True,
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    },
    "traditional": {
        "enabled": True,
        "metrics": ["bleu", "rouge", "meteor"]
    }
}

# Cache settings
CACHE_EXPIRY = 60 * 60 * 24 * 7  # 1 week in seconds
