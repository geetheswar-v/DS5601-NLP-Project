# Advanced RAG System

An advanced RAG (Retrieval Augmented Generation) system built specifically for engineering students. This system leverages local LLM capabilities via Ollama using Granite-3.3 model and Granite-embeddings.

## Features

- Local LLM processing using Ollama with Granite-3.3
- Local embedding generation using Granite-embeddings
- Multi-source web retrieval (Wikipedia, arXiv, scholarly articles, etc.)
- Advanced context refinement techniques
- Concurrent processing for faster retrieval
- Interactive CLI chat mode
- Comprehensive evaluation using RAGAS, BLEU, ROUGE, and METEOR metrics

## Prerequisites

- Python 3.9+
- Ollama installed with Granite-3.3 and Granite-embeddings models
- Poetry for dependency management

## Installation

1. Clone this repository
2. Install dependencies with Poetry:

```bash
cd advanced_rag
poetry install
```

3. Make sure Ollama is running with the required models:

```bash
ollama run granite:3.3
ollama run granite-embeddings
```

## Usage

### Quick Start with run.sh

Use the provided `run.sh` script for easy execution:

```bash
# Install dependencies
./run.sh install

# Start web interface
./run.sh serve

# Start interactive CLI chat
./run.sh cli

# Run a single query
./run.sh query "What is machine learning?"

# Run benchmark
./run.sh benchmark

# Show help
./run.sh help
```

### Web Interface

Start the web interface:

```bash
poetry run python -m advanced_rag.main
# or with logging disabled
poetry run python -m advanced_rag.main --no-logging
```

### CLI Options

Use the CLI for interactive chat or single queries:

```bash
# Interactive chat mode
poetry run python -m advanced_rag.cli chat

# Single query mode
poetry run python -m advanced_rag.cli query "Your question about engineering topics"

# Run benchmark evaluation
poetry run python -m advanced_rag.cli benchmark

# Clear cache
poetry run python -m advanced_rag.cli clear-cache

# Run FastAPI server
poetry run python -m advanced_rag.cli serve
```

Special commands in interactive chat:

- `!help` - Show help
- `!clear` - Clear the screen
- `!logging on` - Enable logging
- `!logging off` - Disable logging
- `exit`, `quit`, or `bye` - Exit the chat

## Evaluation

Run the benchmark evaluation:

```bash
# Run with default questions
poetry run python -m advanced_rag.cli benchmark

# Use a custom dataset
poetry run python -m advanced_rag.cli benchmark --dataset path/to/benchmark.csv
```

The CSV file should have at least a "question" column, and optionally a "reference_answer" column for evaluation.

## License

MIT
