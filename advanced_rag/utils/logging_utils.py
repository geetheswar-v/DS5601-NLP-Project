import logging
import sys
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler

from advanced_rag.config import LOGGING_CONFIG

console = Console()


def setup_logging(
    enable: bool = None,
    level: Union[str, int] = None,
    log_format: str = None,
    log_file: str = None,
) -> None:
    enable = LOGGING_CONFIG["enabled"] if enable is None else enable
    level = level or LOGGING_CONFIG["level"]
    log_format = log_format or LOGGING_CONFIG["format"]
    log_file = log_file or LOGGING_CONFIG["file"]
    
    if not enable:
        logging.disable(logging.CRITICAL)
        return
    
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handlers = []
    
    console_handler = RichHandler(
        rich_tracebacks=True,
        console=console,
        show_time=False,
    )
    handlers.append(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    
    for logger_name in ["httpx", "urllib3"]:
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        logger.setLevel(logging.WARNING)


def enable_logging():
    """Enable logging."""
    logging.disable(logging.NOTSET)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console.print("[green]Logging enabled[/green]")


def disable_logging():
    """Disable all logging."""
    logging.disable(logging.CRITICAL)
    console.print("[yellow]Logging disabled[/yellow]")
