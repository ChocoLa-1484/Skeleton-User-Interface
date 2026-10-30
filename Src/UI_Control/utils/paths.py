"""Project paths independent of the launch directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / 'Src'
PRETRAIN_ROOT = PROJECT_ROOT / 'Db' / 'pretrain'
RECORD_ROOT = PROJECT_ROOT / 'Db' / 'Record'
