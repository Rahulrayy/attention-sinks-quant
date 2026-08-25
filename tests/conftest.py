"""Put the repo root on sys.path so tests import the packages directly.

No src/ layout, no editable install, no packaging config — this repo is read by
humans skimming it in two minutes, not pip-installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
