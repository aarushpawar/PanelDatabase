"""
Core utilities module for Hand Jumper database scraper.

This module provides shared functionality used across all scraper components:
- Configuration management
- Logging setup
- Path resolution
- Common utilities
"""

from .config import Config, load_config
from .logger import setup_logging, get_logger
from .paths import PathManager

__all__ = [
    'Config',
    'load_config',
    'setup_logging',
    'get_logger',
    'PathManager',
]

__version__ = '2.0.0'
