import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from fddim.cli import main


if __name__ == "__main__":
    main()
