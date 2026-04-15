"""vault-memory plugin entry point."""

import os
import sys
from pathlib import Path

# Ensure plugin directory is in path for imports
_plugin_dir = Path(__file__).parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from hermes_plugin import provider  # noqa: F401
