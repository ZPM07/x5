from pathlib import Path

def find_project_root(current_path: Path) -> Path:
    for parent in [current_path, *current_path.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Project root not found")

PROJECT_ROOT = find_project_root(Path(__file__).resolve())
WEIGHTS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'

RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'