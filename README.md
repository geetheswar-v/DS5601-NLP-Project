# DS5601-NLP Project: Context-Aware Science Assistant

This project implements a Retrieval-Augmented Generation (RAG) system designed as a science assistant. It leverages modern Small Language Models (SLMs) via Ollama and runs locally, enabling users to search for information and receive contextually relevant answers.

## Features

- Local LLM processing using Ollama with Granite-3.3 (configurable in `config.py`)
- Local embedding generation using Granite-embeddings (configurable in `config.py`)
- Multi-source web retrieval (Wikipedia, arXiv, scholarly articles, etc.)
- Advanced context refinement techniques
- Concurrent processing for faster retrieval (See in `advanced_rag/retriever`)
- Interactive CLI chat mode used Rich for best interactivatity
- Evaluation Benchmark Using RAGAS & Traditional Metrics
- Benchmarking Models can be changed in (configurable in `config.py`)

## Prerequisites

- Python 3.9+
- Ollama installed and running with the required models (see Configuration)
- Poetry for dependency management

## Installation

1.  Clone this repository:
    ```bash
    git clone <repo_url>
    cd <folder_name>
    ```
2.  Install dependencies using Poetry:
    ```bash
    poetry install
    ```
3.  **Ensure Ollama is running** with the necessary models before starting the application (see Configuration).

## Configuration

Application settings, including the Ollama models used for LLM processing and embeddings, can be configured in the `advanced_rag/config.py` file.

By default, the application uses:
- LLM: `granite:3.3`
- Embeddings: `granite-embeddings`

To use different models:
1.  Make sure the desired models are available in your local Ollama instance (e.g., `ollama pull <model_name>`).
2.  Update the `LLM_MODEL` and `EMBEDDING_MODEL` variables in `advanced_rag/config.py`.

Example Ollama commands to ensure default models are present:
```bash
ollama pull granite:3.3
ollama pull granite-embeddings
# Ensure Ollama server is running (usually starts automatically or via `ollama serve`)
```

## Usage

### Quick Start with run.sh

Use the provided `run.sh` script for easy execution:

```bash
# Make run.sh executable (run once)
chmod +x run.sh

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

The CSV file should have at least a "question" column, and optionally a "reference_answer" column for evaluation. Evaluation results, including RAGAS scores, BLEU, ROUGE, and METEOR metrics (if reference answers are provided), will be saved to a CSV file in the `results` directory.
