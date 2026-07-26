"""Carrega src/config/paths.py sem importar o pacote src (que exige torch)."""

import importlib.util
from pathlib import Path

_paths_file = Path(__file__).resolve().parents[1] / "src" / "config" / "paths.py"
_spec = importlib.util.spec_from_file_location("project_paths", _paths_file)
paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paths)
