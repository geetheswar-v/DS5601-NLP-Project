#!/bin/bash

if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it from https://ollama.ai/"
    exit 1
fi

# Check if required models are available
echo "Checking for required models..."
MODELS=$(ollama list)

# Check for granite:3.3
if ! echo "$MODELS" | grep -q "granite3.3:2b"; then
    echo "Pulling granite3.3 model..."
    ollama pull granite:3.3:2b
else
    echo "Model granite3.3 is already available."
fi

# Check for granite-embeddings
if ! echo "$MODELS" | grep -q "granite-embedding:30m"; then
    echo "Pulling granite-embedding model..."
    ollama pull granite-embedding:30m
else
    echo "Model granite-embedding is already available."
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
if command -v poetry &> /dev/null; then
    poetry install
else
    pip install -e .
fi

# Start the application
echo "Starting the Advanced RAG system..."
python -m advanced_rag.main
