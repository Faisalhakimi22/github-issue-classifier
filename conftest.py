"""Pytest root config — adds project root to sys.path so `from ghic.X import ...`
works regardless of which directory pytest is invoked from."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
