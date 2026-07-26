"""Fusão de representações com varredura de combinações 2→10 por família.

Para cada família em Modelos/ (legado Modelo_Kyoto + as novas Modelo_AE,
Modelo_Skip, Modelo_VAE, Modelo_Joint, Modelo_JointSkip), para cada célula
(base do encoder × dataset de treino × dataset de teste × batch), carrega os
.npy de probabilidade dos modelos disponíveis e avalia:

  - n=1: cada modelo individual (fusion_method "none");
  - n=2..n_disponíveis: combinações (itertools.combinations, amostráveis com
    --frac) com os 5 métodos de fusão (sum, mean_probs, max, mult, vote).

Com --cross, funde 1 modelo de cada família (pool configurável via --pick).

Ground truth validado por contagem de linhas (protocolo legado: teste==treino
usa {ds}_test.csv, senão {ds}.csv) — aborta a célula se nada casar.

Retomável: células já presentes no CSV de saída são puladas.

Uso:
    python fusion_families.py [--families Modelo_AE,...] [--frac 0.5] [--cross]
"""

import argparse
import logging
import multiprocessing
import random
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from fusion import fusion_max, fusion_mult, fusion_sum, fusion_vote

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELOS_DIR = PROJECT_ROOT / "Modelos"
CSV_DIR = PROJECT_ROOT / "CSV"

FAMILY_SPECS = {
    "Modelo_Kyoto": 10,
    "Modelo_AE": 10,
    "Modelo_Skip": 10,
    "Modelo_VAE": 10,
    "Modelo_Joint": 2,
    "Modelo_JointSkip": 2,
}

CAMERAS = [f"camera{i}" for i in range(1, 10)]
PKLOT_SETS = ["PUC", "UFPR04", "UFPR05"]

ENCODER_TO_TRAINS = {
    "CNR": PKLOT_SETS,
    "PKLot": CAMERAS,
    "Kyoto": PKLOT_SETS + CAMERAS,
}
TESTES = PKLOT_SETS + CAMERAS
BATCHES = [64, 128, 256, 512, 1024]

SEED = 42

COLUMNS = [
    "tipo_encoder", "dataset_encoder", "dataset_train", "dataset_test", "batch",
    "n_models", "combination", "tecnica_fusao",
    "precision", "accuracy", "f1", "recall", "auc",
]


def npy_path(tipo, model_id, base, train, test, batch):
    return (MODELOS_DIR / f"{tipo}-{model_id}" / f"Classificador-{base}"
            / "Resultados" / f"Treinados_em_{train}" / test / f"batches-{batch}.npy")


def load_y_true(test, n_samples):
    """Escolhe o CSV de labels pelo número de linhas (padrão do ablation.py)."""
    for suffix in ("_test.csv", ".csv"):
        p = CSV_DIR / test / f"{test}{suffix}"
        if p.is_file():
            y = pd.read_csv(p)["class"].values
            if len(y) == n_samples:
                return y.astype(np.int64)
    raise ValueError(f"Nenhum CSV de labels com {n_samples} linhas para '{test}'")


def compute_metrics(y_true, probs):
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs[:, 1])
    except ValueError:
        auc = float("nan")
    return {
        "precision": round(prec, 6),
        "accuracy": round(acc, 6),
        "f1": round(f1, 6),
        "recall": round(rec, 6),
        "auc": round(auc, 6) if not np.isnan(auc) else "nan",
    }


def load_cell_arrays(members, base, train, test, batch):
    """members: lista de (tipo, model_id). Retorna (labels, arrays) dos existentes."""
    labels, arrays = [], []
    n_rows = None
    for tipo, model_id in members:
        p = npy_path(tipo, model_id, base, train, test, batch)
        if not p.is_file():
            continue
        arr = np.load(p).astype(np.float32, copy=False)
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError(f"shape inconsistente em {p}: {arr.shape[0]} != {n_rows}")
        labels.append((tipo, model_id))
        arrays.append(arr)
    return labels, arrays


def evaluate_combinations(base_row, labels, arrays, y_true, frac, min_n, max_n,
                          qualified_names):
    """Gera as linhas de métricas para todas as combinações de uma célula."""
    rows = []
    n_available = len(arrays)
    stacked_all = np.stack(arrays, axis=0)

    def comb_name(combo):
        if qualified_names:
            return str(tuple(f"{labels[i][0]}-{labels[i][1]}" for i in combo))
        return str(tuple(labels[i][1] for i in combo))

    if min_n <= 1:
        for i in range(n_available):
            rows.append({
                **base_row, "n_models": 1, "combination": comb_name((i,)),
                "tecnica_fusao": "none", **compute_metrics(y_true, arrays[i]),
            })

    for n in range(max(2, min_n), min(n_available, max_n) + 1):
        all_combos = list(combinations(range(n_available), n))
        if frac < 1.0:
            # Seed derivada da célula: amostra reprodutível mesmo com retomada
            # (crc32 é estável entre processos, hash() de str não é).
            rng = random.Random(zlib.crc32(f"{SEED}|{sorted(base_row.items())}|{n}".encode()))
            all_combos = rng.sample(all_combos, max(1, int(len(all_combos) * frac)))

        for combo in all_combos:
            stacked = stacked_all[list(combo)]
            name = comb_name(combo)

            # sum e mean_probs produzem as mesmas probabilidades normalizadas:
            # computa uma vez e emite as duas linhas.
            metrics_sum = compute_metrics(y_true, fusion_sum(stacked))
            for method in ("sum", "mean_probs"):
                rows.append({
                    **base_row, "n_models": n, "combination": name,
                    "tecnica_fusao": method, **metrics_sum,
                })
            for method, func in (("max", fusion_max), ("mult", fusion_mult),
                                 ("vote", fusion_vote)):
                rows.append({
                    **base_row, "n_models": n, "combination": name,
                    "tecnica_fusao": method, **compute_metrics(y_true, func(stacked)),
                })
    return rows


def process_cell(args):
    (tipo, members, base, train, test, batch, frac, min_n, max_n, qualified) = args
    try:
        labels, arrays = load_cell_arrays(members, base, train, test, batch)
        if len(arrays) < 1 or (len(arrays) < 2 and min_n >= 2):
            return []
        y_true = load_y_true(test, arrays[0].shape[0])
        base_row = {
            "tipo_encoder": tipo,
            "dataset_encoder": base,
            "dataset_train": train,
            "dataset_test": test,
            "batch": batch,
        }
        return evaluate_combinations(base_row, labels, arrays, y_true,
                                     frac, min_n, max_n, qualified)
    except Exception as e:
        logger.warning(f"[célula pulada] {tipo}/{base}/{train}/{test}/b{batch}: {e}")
        return []


def cell_key(row_or_tuple):
    if isinstance(row_or_tuple, dict):
        return (row_or_tuple["tipo_encoder"], row_or_tuple["dataset_encoder"],
                row_or_tuple["dataset_train"], row_or_tuple["dataset_test"],
                str(row_or_tuple["batch"]))
    return tuple(str(x) for x in row_or_tuple)


def load_done_cells(out_path):
    if not out_path.is_file():
        return set()
    df = pd.read_csv(out_path, usecols=["tipo_encoder", "dataset_encoder",
                                        "dataset_train", "dataset_test", "batch"])
    return {cell_key(t) for t in df.itertuples(index=False, name=None)}


def build_tasks(args):
    tasks = []
    families = ([f.strip() for f in args.families.split(",")]
                if args.families != "all" else list(FAMILY_SPECS))
    for f in families:
        if f not in FAMILY_SPECS:
            raise ValueError(f"Família inválida: {f}. Use {list(FAMILY_SPECS)}")

    bases = ([b.strip() for b in args.bases.split(",")]
             if args.bases != "all" else list(ENCODER_TO_TRAINS))
    batches = ([int(b) for b in args.batches.split(",")]
               if args.batches != "all" else BATCHES)

    if args.cross:
        pool = parse_pick(args.pick, families)
        for base in bases:
            for train in ENCODER_TO_TRAINS[base]:
                for test in TESTES:
                    for batch in batches:
                        tasks.append(("cross", pool, base, train, test, batch,
                                      args.frac, args.min_n, args.max_n, True))
        return tasks

    for tipo in families:
        members = [(tipo, i) for i in range(FAMILY_SPECS[tipo])]
        for base in bases:
            for train in ENCODER_TO_TRAINS[base]:
                for test in TESTES:
                    for batch in batches:
                        tasks.append((tipo, members, base, train, test, batch,
                                      args.frac, args.min_n, args.max_n, False))
    return tasks


def parse_pick(pick, families):
    """--pick "Modelo_AE:3,Modelo_Skip:0" -> pool de 1 modelo por família."""
    chosen = {tipo: 0 for tipo in families}
    if pick:
        for item in pick.split(","):
            tipo, _, idx = item.partition(":")
            if tipo not in FAMILY_SPECS:
                raise ValueError(f"--pick com família inválida: {tipo}")
            chosen[tipo] = int(idx)
    return [(tipo, idx) for tipo, idx in chosen.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=str, default="all",
                        help=f"all ou lista: {','.join(FAMILY_SPECS)}")
    parser.add_argument("--bases", type=str, default="all", help="all | CNR,PKLot,Kyoto")
    parser.add_argument("--batches", type=str, default="all", help="all | 64,128,...")
    parser.add_argument("--frac", type=float, default=1.0,
                        help="Fração das combinações a avaliar (amostra com seed fixa)")
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--cross", action="store_true",
                        help="Fusão cross-família: 1 modelo por família (ver --pick)")
    parser.add_argument("--pick", type=str, default="",
                        help='Índice por família no modo --cross, ex.: "Modelo_AE:3,Modelo_VAE:1"')
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--out", type=str, default="",
                        help="CSV de saída (default fusion_families.csv ou fusion_families_cross.csv)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / ("fusion_families_cross.csv" if args.cross else "fusion_families.csv")
    )

    tasks = build_tasks(args)
    done = load_done_cells(out_path)
    if done:
        before = len(tasks)
        tasks = [t for t in tasks
                 if cell_key((t[0], t[2], t[3], t[4], t[5])) not in done]
        logger.info(f"Retomada: {before - len(tasks)} células já no CSV, {len(tasks)} restantes")

    logger.info(f"Células a processar: {len(tasks)} | saída: {out_path}")
    if not tasks:
        return

    write_header = not out_path.is_file()
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_cell, t): t for t in tasks}
        with open(out_path, "a", newline="") as f:
            for fut in as_completed(futures):
                rows = fut.result()
                completed += 1
                if rows:
                    df = pd.DataFrame(rows, columns=COLUMNS)
                    df.to_csv(f, header=write_header, index=False)
                    f.flush()
                    write_header = False
                if completed % 25 == 0 or completed == len(tasks):
                    logger.info(f"[{completed}/{len(tasks)}] células processadas")

    logger.info(f"Concluído → {out_path}")


if __name__ == "__main__":
    main()
