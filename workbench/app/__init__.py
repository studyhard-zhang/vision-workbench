"""Single-user visual training workspace."""
import sys
from pathlib import Path

vendor = Path(__file__).resolve().parent.parent / '.vendor'
if vendor.is_dir():
    sys.path.insert(0, str(vendor))

