"""Verifica o alinhamento entre os .npy legados e os CSVs de teste.

Protocolo de teste do legado (confirmado contra todos os .npy de
Modelos/Modelo_Kyoto-*):
  - teste == treino  -> avaliado no split  {ds}_test.csv  (versão de CSV_antigo)
  - teste != treino  -> avaliado no CSV completo {ds}.csv
Os .npy seguem a ORDEM de linhas desses CSVs da máquina antiga (CSV_antigo/).

Este script confere, para cada dataset de teste, se CSV/{ds}/{ds}_test.csv e
CSV/{ds}/{ds}.csv têm o mesmo conteúdo e ordem das versões de CSV_antigo; com
--canonize, substitui os divergentes (com backup), garantindo que a nova
geração de modelos avalie na mesma ordem do legado.

Uso:
    python scripts/verify_test_alignment.py [--canonize]
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import paths

PROJECT_ROOT, CSV_DIR, CSV_ANTIGO_DIR = (
    paths.PROJECT_ROOT, paths.CSV_DIR, paths.CSV_ANTIGO_DIR
)

BACKUP_DIR = PROJECT_ROOT / "CSV_backup_paths"

TESTES = [
    "PUC", "UFPR04", "UFPR05",
    "camera1", "camera2", "camera3", "camera4", "camera5",
    "camera6", "camera7", "camera8", "camera9",
]


def relative_key(series: pd.Series) -> pd.Series:
    # Chave independente do prefixo da máquina: caminho após a raiz do dataset.
    for marker in ("PKLotSegmented/", "PATCHES/"):
        if series.iloc[0].find(marker) != -1:
            return series.str.split(marker, n=1).str[-1]
    return series.str.rsplit("/", n=1).str[-1]


def compare(atual_path: Path, antigo_path: Path):
    if not antigo_path.is_file():
        return "sem_antigo"
    if not atual_path.is_file():
        return "faltando"
    atual = pd.read_csv(atual_path)
    antigo = pd.read_csv(antigo_path)
    if len(atual) != len(antigo):
        return "diverge"
    if not relative_key(atual["path_image"]).equals(relative_key(antigo["path_image"])):
        return "diverge"
    return "ok"


def canonize(atual_path: Path, antigo_path: Path):
    backup = BACKUP_DIR / "CSV" / atual_path.relative_to(CSV_DIR)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if atual_path.is_file() and not backup.exists():
        shutil.copy2(atual_path, backup)
    shutil.copy2(antigo_path, atual_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonize", action="store_true",
                        help="Substitui CSVs divergentes pela versão de CSV_antigo (com backup)")
    args = parser.parse_args()

    problems = 0
    for test in TESTES:
        for suffix in (f"{test}_test.csv", f"{test}.csv"):
            atual_path = CSV_DIR / test / suffix
            antigo_path = CSV_ANTIGO_DIR / test / suffix
            status = compare(atual_path, antigo_path)
            print(f"{test:9s} {suffix:22s} [{status}]")
            if status != "diverge":
                continue
            problems += 1
            if args.canonize:
                canonize(atual_path, antigo_path)
                print(f"  -> canonizado a partir de {antigo_path.relative_to(PROJECT_ROOT)}")
            else:
                print("  -> rode com --canonize para copiar a versão de CSV_antigo")

    if problems == 0:
        print("\nTodos os CSVs de teste alinhados com o legado.")
    else:
        print(f"\n{problems} CSV(s) divergente(s)"
              + (" — canonizados." if args.canonize else "."))
        if not args.canonize:
            sys.exit(1)


if __name__ == "__main__":
    main()
