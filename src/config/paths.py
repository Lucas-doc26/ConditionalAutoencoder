import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_DIR = PROJECT_ROOT / "CSV"
CSV_ANTIGO_DIR = PROJECT_ROOT / "CSV_antigo"
MODELOS_DIR = PROJECT_ROOT / "Modelos"

DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", str(Path.home() / "DeepLearning" / "datasets")))


def _dataset_dir(env_var: str, *candidates: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    for name in candidates:
        candidate = DATASETS_ROOT / name
        if candidate.is_dir():
            return candidate
    return DATASETS_ROOT / candidates[0]


CNR_DIR = _dataset_dir("CNR_DIR", "CNR")
PKLOT_DIR = _dataset_dir("PKLOT_DIR", "PKLot")
KYOTO_DIR = _dataset_dir("KYOTO_DIR", "Kyoto", "kyoto")
