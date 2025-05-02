import os
from pathlib import Path

# Package-level constants
PKG_DIR = Path(__file__).parent

# Import key classes
from .handler import Preprocessor, Fetcher  # noqa: F401
from .recommend import Recommend  # noqa: F401

__version__ = '0.1'