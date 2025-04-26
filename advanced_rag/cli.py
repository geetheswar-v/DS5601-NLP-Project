import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from advanced_rag.config import CLI_CONFIG, LOGGING_CONFIG, SOURCES
from advanced_rag.database.vector_store import VectorStore
from advanced_rag.evaluation.benchmark import RagBenchmark
from advanced_rag.evaluation.metrics import RagEvaluator
from advanced_rag.llm.ollama_client import OllamaClient
from advanced_rag.models.document import SourceType
from advanced_rag.rag.pipeline import RagPipeline
from advanced_rag.retrieval.web_retrieval import WebRetriever
from advanced_rag.utils.cache_manager import CacheManager
from advanced_rag.utils.logging_utils import setup_logging, enable_logging, disable_logging
from advanced_rag.utils.text_processor import TextProcessor

# Initialize Typer app
app = typer.Typer(help="ContentAware App System(ADV RAG) CLI")
console = Console()
chat_history = []


def setup_rag_pipeline() -> RagPipeline:
    ollama_client = OllamaClient()
    cache_manager = CacheManager()
    text_processor = TextProcessor()
    
    web_retriever = WebRetriever(cache_manager=cache_manager, text_processor=text_processor)
    
    vector_store = VectorStore(
        ollama_client=ollama_client,
        text_processor=text_processor,
    )
    
    pipeline = RagPipeline(
        ollama_client=ollama_client,
        web_retriever=web_retriever,
        vector_store=vector_store,
        text_processor=text_processor,
        cache_manager=cache_manager,
    )
    
    return pipeline


@app.command("query")
def process_query(
    query: str = typer.Argument(..., help="The query to process"),
    use_web: bool = typer.Option(CLI_CONFIG["use_web"], "--web/--no-web", help="Whether to use web search"),
    refresh_cache: bool = typer.Option(False, help="Whether to refresh the cache"),
):
    pipeline = setup_rag_pipeline()
    
    with console.status("[bold green]Processing query..."):
        answer, docs = asyncio.run(pipeline.process_query(
            query=query,
            generate_answer=True,
            use_web=use_web,
            refresh_cache=refresh_cache,
        ))
    
    # Display answer
    console.print(Panel(answer, title="Answer", border_style="green"))
    
    # Display sources
    if docs:
        table = Table(title="Sources")
        table.add_column("Source", style="cyan")
        table.add_column("Title")
        table.add_column("URL", style="blue")
        table.add_column("Relevance", justify="right")
        
        for doc in docs:
            relevance = f"{doc.metadata.relevance_score*100:.1f}%" if doc.metadata.relevance_score else "N/A"
            table.add_row(
                str(doc.metadata.source_type.value) if hasattr(doc.metadata.source_type, "value") else str(doc.metadata.source_type),
                doc.metadata.title,
                doc.metadata.url or "N/A",
                relevance,
            )
        
        console.print(table)


@app.command("chat")
def interactive_chat(
    use_web: bool = typer.Option(CLI_CONFIG["use_web"], "--web/--no-web", help="Whether to use web search"),
    refresh_cache: bool = typer.Option(False, help="Whether to refresh the cache"),
):
    """Start an interactive chat session with the RAG system."""
    pipeline = setup_rag_pipeline()
    
    console.print(Panel(
        "Welcome to the ContextAware! Type 'exit', 'quit', or press Ctrl+C to exit., For help, type !help",
        title="Interactive Chat",
        border_style="blue",
    ))
    
    # Keep a history of conversation for context
    history = []
    
    try:
        while True:
            # Get user query
            query = Prompt.ask("\n[bold green]You[/bold green]")
            
            # Check exit commands
            if query.lower() in ["exit", "quit", "bye"]:
                console.print("[yellow]Exiting chat...[/yellow]")
                break
            
            # Process special commands
            if query.startswith("!"):
                handle_special_command(query)
                continue
            
            # Add to history
            history.append({"role": "user", "content": query})
            
            # Process the query
            with console.status("[bold green]Processing query..."):
                answer, docs = asyncio.run(pipeline.process_query(
                    query=query,
                    generate_answer=True,
                    use_web=use_web,
                    refresh_cache=refresh_cache,
                ))
            
            # Add to history (limit size if needed)
            history.append({"role": "assistant", "content": answer})
            if len(history) > CLI_CONFIG["history_size"] * 2:  # *2 because each exchange has 2 entries
                history = history[-CLI_CONFIG["history_size"]*2:]
            
            # Display answer
            console.print("\n[bold blue]Assistant[/bold blue]")
            console.print(Panel(answer, border_style="blue"))
            
            # Display sources
            if docs:
                table = Table(title="Sources")
                table.add_column("Source", style="cyan")
                table.add_column("Title")
                table.add_column("URL", style="blue")
                
                for doc in docs:
                    table.add_row(
                        str(doc.metadata.source_type.value) if hasattr(doc.metadata.source_type, "value") else str(doc.metadata.source_type),
                        doc.metadata.title or "Unknown",
                        doc.metadata.url or "N/A",
                    )
                
                console.print(table)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat session terminated by user.[/yellow]")
    
    console.print("[green]Chat session ended. Goodbye![/green]")


def handle_special_command(command: str):
    cmd = command.strip().lower()
    
    if cmd == "!help":
        console.print(Panel(
            "Available commands:\n"
            "!help - Show this help message\n"
            "!clear - Clear the screen\n"
            "!logging on - Enable logging\n"
            "!logging off - Disable logging\n"
            "exit, quit, bye - Exit the chat",
            title="Chat Commands",
            border_style="yellow",
        ))
    elif cmd == "!clear":
        os.system('cls' if os.name == 'nt' else 'clear')
    elif cmd == "!logging on":
        enable_logging()
    elif cmd == "!logging off":
        disable_logging()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Type !help for available commands")


@app.command("benchmark")
def run_benchmark(
    dataset: Path = typer.Option(None, help="Path to benchmark dataset CSV"),
    output: Path = typer.Option(None, help="Path to save benchmark results"),
    use_web: bool = typer.Option(True, help="Whether to use web search"),
    num_questions: int = typer.Option(10, help="Number of questions for new dataset"),
):
    """Run benchmark evaluation with optional CSV input."""
    pipeline = setup_rag_pipeline()
    evaluator = RagEvaluator()
    benchmark = RagBenchmark(pipeline, evaluator)
    
    # Validate dataset if provided
    if dataset and not dataset.exists():
        console.print(f"[red]Error: Dataset file {dataset} not found[/red]")
        raise typer.Exit(1)
    
    with console.status("[bold green]Running benchmark..."):
        results, df = asyncio.run(benchmark.run_benchmark(
            dataset_path=dataset,
            use_web=use_web,
            num_questions=num_questions,
            output_path=output
        ))
    
    # Display summary
    console.print(Panel(
        f"Processed {len(df)} questions\n" +
        (f"Results saved to: {output}" if output else ""),
        title="Benchmark Summary", 
        border_style="blue"
    ))
    
    # Display metrics if available
    if "traditional" in results and "ragas" in results:
        console.print("[bold green]Evaluation Metrics:[/bold green]")
        
        if "traditional" in results:
            trad = results["traditional"]
            console.print("\n[bold]Traditional Metrics:[/bold]")
            for metric, value in trad.items():
                if isinstance(value, float):
                    console.print(f"  {metric}: {value:.4f}")
        
        if "ragas" in results:
            ragas = results["ragas"]
            console.print("\n[bold]RAGAS Metrics:[/bold]")
            for metric, value in ragas.items():
                if isinstance(value, float):
                    console.print(f"  {metric}: {value:.4f}")
    else:
        console.print("[yellow]No evaluation metrics available. Add reference answers to the dataset for evaluation.[/yellow]")


@app.command("clear-cache")
def clear_cache():
    """Clear the cache."""
    cache_manager = CacheManager()
    cache_manager.clear()
    console.print("[green]Cache cleared successfully![/green]")


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server"),
    port: int = typer.Option(8000, help="Port to bind the server"),
    logging_enabled: bool = typer.Option(True, "--logging/--no-logging", help="Enable or disable logging"),
):
    """Serve the web UI."""
    # Set logging based on parameter
    if not logging_enabled:
        disable_logging()
    
    import uvicorn
    from advanced_rag.ui.api import RagAPI
    
    pipeline = setup_rag_pipeline()
    api = RagAPI(pipeline)
    app = api.get_app()
    
    console.print(f"[green]Starting server at http://{host}:{port}[/green]")
    uvicorn.run(app, host=host, port=port, log_level="error" if not logging_enabled else "info")


@app.callback()
def main(
    logging_enabled: bool = typer.Option(
        LOGGING_CONFIG["enabled"], 
        "--logging/--no-logging", 
        help="Enable or disable logging"
    ),
    log_level: str = typer.Option(
        LOGGING_CONFIG["level"],
        "--log-level",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ),
):
    setup_logging(
        enable=logging_enabled,
        level=log_level,
    )


if __name__ == "__main__":
    app()
