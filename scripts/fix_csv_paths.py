"""Reescreve os prefixos antigos de path_image nos CSVs para os datasets da máquina atual.

Os CSVs do repositório foram gerados em outra máquina e apontam para
/home/lucas.ocunha/... . Este script mapeia os prefixos conhecidos para os
diretórios definidos em src/config/paths.py (configuráveis via DATASETS_ROOT,
CNR_DIR, PKLOT_DIR, KYOTO_DIR), sem nunca reordenar ou filtrar linhas.

Uso:
    python scripts/fix_csv_paths.py [--dirs CSV CSV_antigo] [--map ANTIGO=NOVO ...]
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import paths

PROJECT_ROOT, CNR_DIR, PKLOT_DIR, KYOTO_DIR = (
    paths.PROJECT_ROOT, paths.CNR_DIR, paths.PKLOT_DIR, paths.KYOTO_DIR
)

BACKUP_DIR = PROJECT_ROOT / "CSV_backup_paths"
SAMPLE_SIZE = 30


def default_mappings():
    return [
        ("/home/lucas.ocunha/ConditionalAutoencoder/PKLot", str(PKLOT_DIR)),
        ("/home/lucas.ocunha/DeepLearning/datasets/PKLot", str(PKLOT_DIR)),
        ("/home/lucas.ocunha/DeepLearning/datasets/CNR", str(CNR_DIR)),
        ("/home/lucas.ocunha/DeepLearning/datasets/Kyoto", str(KYOTO_DIR)),
        ("/datasets/CNR-EXT-Patches-150x150", str(CNR_DIR)),
        ("/datasets/PKLot", str(PKLOT_DIR)),
        ("/datasets/Kyoto", str(KYOTO_DIR)),
    ]


def rewrite_csv(csv_path: Path, mappings, root: Path) -> bool:
    df = pd.read_csv(csv_path)
    path_col = "path_image" if "path_image" in df.columns else df.columns[0]

    if df.empty or not str(df[path_col].iloc[0]).startswith("/"):
        return False

    original = df[path_col].copy()
    for old, new in mappings:
        df[path_col] = df[path_col].str.replace(old, new, regex=False)

    if df[path_col].equals(original):
        return False

    backup_path = BACKUP_DIR / csv_path.relative_to(root)
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, backup_path)

    df.to_csv(csv_path, index=False)
    return True


def sample_existing(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    path_col = "path_image" if "path_image" in df.columns else df.columns[0]
    paths = df[path_col].dropna().tolist()
    if not paths:
        return 0.0
    sample = random.Random(42).sample(paths, min(SAMPLE_SIZE, len(paths)))
    return 100.0 * sum(os.path.exists(p) for p in sample) / len(sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", default=["CSV", "CSV_antigo"],
                        help="Diretórios (relativos à raiz do projeto) a varrer")
    parser.add_argument("--map", action="append", default=[], metavar="ANTIGO=NOVO",
                        help="Mapeamento extra de prefixo (pode repetir)")
    args = parser.parse_args()

    mappings = default_mappings()
    for extra in args.map:
        old, _, new = extra.partition("=")
        if not new:
            parser.error(f"--map inválido: {extra} (esperado ANTIGO=NOVO)")
        mappings.insert(0, (old, new))

    print("Mapeamentos:")
    for old, new in mappings:
        print(f"  {old} -> {new}")

    changed = 0
    report = []
    for rel_dir in args.dirs:
        root = PROJECT_ROOT / rel_dir
        if not root.is_dir():
            print(f"[aviso] diretório inexistente: {root}")
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            if rewrite_csv(csv_path, mappings, root):
                changed += 1
            report.append((csv_path, sample_existing(csv_path)))

    print(f"\n{changed} CSVs reescritos (backup em {BACKUP_DIR}).")
    print("\n% de paths existentes (amostra de até "
          f"{SAMPLE_SIZE} linhas por CSV):")
    missing = [(p, pct) for p, pct in report if pct < 100.0]
    for csv_path, pct in missing:
        print(f"  {pct:5.1f}%  {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  ({len(report) - len(missing)} CSVs com 100% dos paths existentes)")


if __name__ == "__main__":
    main()
