#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Poetry not found. Please install Poetry first:${NC}"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if we're inside the virtual environment or need to use poetry run
IN_VENV=false
if [[ -n "$VIRTUAL_ENV" ]]; then
    IN_VENV=true
fi

run_command() {
    if $IN_VENV; then
        eval "$@"
    else
        poetry run "$@"
    fi
}

show_help() {
    echo -e "${BLUE}Advanced RAG System Runner${NC}"
    echo 
    echo "Usage: ./run.sh [command] [options]"
    echo 
    echo "Commands:"
    echo "  serve       Start the web UI"
    echo "  cli         Start the CLI interactive chat"
    echo "  query       Run a single query in CLI mode"
    echo "  benchmark   Run benchmark evaluation"
    echo "  install     Install dependencies with Poetry"
    echo "  help        Show this help message"
    echo
    echo "Options:"
    echo "  --no-logging    Disable logging for any command"
    echo "  --port PORT     Specify port for the server (default: 8000)"
    echo "  --host HOST     Specify host for the server (default: 0.0.0.0)"
    echo 
    echo "Examples:"
    echo "  ./run.sh serve"
    echo "  ./run.sh serve --no-logging --port 9000"
    echo "  ./run.sh cli"
    echo "  ./run.sh query \"What is machine learning?\""
    echo "  ./run.sh benchmark --dataset path/to/benchmark.csv"
}

# Function to check if Ollama is running
check_ollama() {
    echo -e "${BLUE}Checking if Ollama is running...${NC}"
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo -e "${RED}Ollama is not running. Please start Ollama first:${NC}"
        echo "ollama serve"
        exit 1
    else
        echo -e "${GREEN}Ollama is running.${NC}"
    fi
}

# Parse command
COMMAND=$1
shift || true

case $COMMAND in
    serve)
        check_ollama
        echo -e "${GREEN}Starting FastAPI server...${NC}"
        run_command "python -m advanced_rag.main $*"
        ;;
        
    cli)
        check_ollama
        echo -e "${GREEN}Starting interactive CLI chat...${NC}"
        run_command "python -m advanced_rag.cli chat $*"
        ;;
        
    query)
        check_ollama
        if [ -z "$1" ]; then
            echo -e "${RED}Error: No query provided${NC}"
            echo "Usage: ./run.sh query \"your question here\""
            exit 1
        fi
        
        echo -e "${GREEN}Processing query: $1${NC}"
        run_command "python -m advanced_rag.cli query \"$1\" ${@:2}"
        ;;
        
    benchmark)
        check_ollama
        echo -e "${GREEN}Running benchmark...${NC}"
        run_command "python -m advanced_rag.cli benchmark $*"
        ;;
        
    install)
        echo -e "${GREEN}Installing dependencies with Poetry...${NC}"
        poetry install
        echo -e "${BLUE}Installing NLTK data...${NC}"
        run_command "python -c \"import nltk; nltk.download('punkt'); nltk.download('wordnet')\""
        echo -e "${GREEN}Installation complete!${NC}"
        ;;
        
    help|--help|-h)
        show_help
        ;;
        
    *)
        echo -e "${RED}Error: Unknown command '$COMMAND'${NC}"
        show_help
        exit 1
        ;;
esac
