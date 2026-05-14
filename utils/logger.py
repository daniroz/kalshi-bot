"""Rich console + file logger."""

import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True, markup=True),
        logging.FileHandler("kalshi_bot.log"),
    ],
)

log = logging.getLogger("kalshi_bot")
