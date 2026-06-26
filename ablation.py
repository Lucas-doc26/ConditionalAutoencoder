import os
import random
import logging
import multiprocessing
import threading
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
)
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# Paths
# =========================================================

BASE_MODEL_DIR = "/home/lucas/representation-fusion/Modelos"
BASE_CSV_DIR   = "/home/lucas/representation-fusion/CSV"
OUT_DIR        = "/home/lucas/representation-fusion/ablation_results"

# =========================================================
# Constants
# =========================================================

N_MODELS       = 10
ALL_MODEL_IDS  = list(range(N_MODELS))
BATCHES        = [64, 128, 256, 512, 1024]

DICT_COMBINATIONS = {
    "PKLot": ["camera1", "camera2", "camera3", "camera4", "camera5",
              "camera6", "camera7", "camera8", "camera9"],
    "CNR":   ["UFPR04", "UFPR05", "PUC"],
    "Kyoto": ["UFPR04", "UFPR05", "PUC",
              "camera1", "camera2", "camera3", "camera4", "camera5",
              "camera6", "camera7", "camera8", "camera9"],
}

TESTES = [
    "UFPR04", "UFPR05", "PUC",
    "camera1", "camera2", "camera3", "camera4", "camera5",
    "camera6", "camera7", "camera8", "camera9",
]

# =========================================================
# Fusion methods
# =========================================================

def fusion_sum(stacked):
    agg = np.sum(stacked, axis=0)
    return agg / agg.sum(axis=1, keepdims=True)


def fusion_mean(stacked):
    agg = np.mean(stacked, axis=0)
    return agg / agg.sum(axis=1, keepdims=True)


def fusion_max(stacked):
    agg = np.max(stacked, axis=0)
    return agg / agg.sum(axis=1, keepdims=True)


def fusion_mult(stacked):
    logp = np.log(np.clip(stacked, 1e-300, 1.0))
    sumlog = np.sum(logp, axis=0)
    exp = np.exp(sumlog - np.max(sumlog, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def fusion_vote(stacked):
    n_models, N, C = stacked.shape
    preds_models = np.argmax(stacked, axis=2)
    final_preds = np.zeros(N, dtype=np.int64)

    for i in range(N):
        votes = preds_models[:, i]
        counts = Counter(votes)
        top = counts.most_common()
        max_votes = top[0][1]
        tied = [cls for cls, cnt in top if cnt == max_votes]
        if len(tied) == 1:
            final_preds[i] = tied[0]
        else:
            probs_sum = np.sum(stacked[:, i, :], axis=0)
            final_preds[i] = max(tied, key=lambda c: probs_sum[c])

    return np.eye(C, dtype=np.float32)[final_preds]


FUSION_METHODS = {
    "sum":        fusion_sum,
    "mean_probs": fusion_mean,
    "max":        fusion_max,
    "mult":       fusion_mult,
    "vote":       fusion_vote,
}

# =========================================================
# Helpers
# =========================================================

def npy_path(encoder, classificador, teste, batch, model_id):
    return (
        f"{BASE_MODEL_DIR}/Modelo_Kyoto-{model_id}/"
        f"Classificador-{encoder}/Resultados/"
        f"Treinados_em_{classificador}/{teste}/batches-{batch}.npy"
    )


def load_arrays(encoder, classificador, teste, batch):
    arrays = {}
    n_samples_set = set()
    for mid in ALL_MODEL_IDS:
        p = npy_path(encoder, classificador, teste, batch, mid)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing: {p}")
        arr = np.load(p).astype(np.float32, copy=False)
        arrays[mid] = arr
        n_samples_set.add(arr.shape[0])
    if len(n_samples_set) != 1:
        raise ValueError(f"Inconsistent sample counts: {n_samples_set}")
    return arrays, next(iter(n_samples_set))


def load_y_true(teste, n_samples):
    for suffix in ["_test.csv", ".csv"]:
        p = f"{BASE_CSV_DIR}/{teste}/{teste}{suffix}"
        if os.path.isfile(p):
            y = pd.read_csv(p)["class"].values
            if len(y) == n_samples:
                return y.astype(np.int64)
    raise ValueError(f"No label CSV with {n_samples} rows for '{teste}'")


def compute_metrics(y_true, y_pred, probs):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs[:, 1])
    except ValueError:
        auc = float("nan")
    return acc, prec, f1, rec, auc


def sample_combinations(all_combs, frac=0.5):
    k = max(1, int(len(all_combs) * frac))
    return random.sample(all_combs, k)


# =========================================================
# Worker
# =========================================================

def process_task(args):
    encoder, classificador, teste, batch = args
    rows = []

    ref_dir = (
        f"{BASE_MODEL_DIR}/Modelo_Kyoto-0/"
        f"Classificador-{encoder}/Resultados/"
        f"Treinados_em_{classificador}/{teste}"
    )
    if not os.path.isdir(ref_dir):
        return rows

    try:
        arrays, n_samples = load_arrays(encoder, classificador, teste, batch)
    except Exception as e:
        logger.warning(f"[skip load] {encoder}/{classificador}/{teste}/b{batch}: {e}")
        return rows

    try:
        y_true = load_y_true(teste, n_samples)
    except Exception as e:
        logger.warning(f"[skip labels] {encoder}/{classificador}/{teste}/b{batch}: {e}")
        return rows

    base = {
        "encoder":       encoder,
        "classificador": classificador,
        "teste":         teste,
        "batch":         batch,
    }

    # --- n = 1: individual models (no fusion) ---
    for mid in ALL_MODEL_IDS:
        probs  = arrays[mid]
        y_pred = np.argmax(probs, axis=1)
        acc, prec, f1, rec, auc = compute_metrics(y_true, y_pred, probs)
        rows.append({
            **base,
            "n_models":      1,
            "combination":   (mid,),
            "fusion_method": "none",
            "accuracy":      acc,
            "precision":     prec,
            "f1":            f1,
            "recall":        rec,
            "auc":           auc,
        })

    # --- n = 2 .. 10: ensemble with fusion ---
    for n in range(2, N_MODELS + 1):
        all_combs = list(combinations(ALL_MODEL_IDS, n))
        selected  = sample_combinations(all_combs, frac=0.5)

        for comb in selected:
            stacked = np.stack([arrays[mid] for mid in comb], axis=0)

            for fname, ffunc in FUSION_METHODS.items():
                fused  = ffunc(stacked)
                y_pred = np.argmax(fused, axis=1)
                acc, prec, f1, rec, auc = compute_metrics(y_true, y_pred, fused)
                rows.append({
                    **base,
                    "n_models":      n,
                    "combination":   comb,
                    "fusion_method": fname,
                    "accuracy":      acc,
                    "precision":     prec,
                    "f1":            f1,
                    "recall":        rec,
                    "auc":           auc,
                })

    return rows


# =========================================================
# Main
# =========================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tasks = [
        (encoder, classificador, teste, batch)
        for encoder, classificadores in DICT_COMBINATIONS.items()
        for classificador in classificadores
        for teste in TESTES
        for batch in BATCHES
    ]
    logger.info(f"Total tasks: {len(tasks)}")

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {n_workers} worker processes")

    all_rows = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_task, t): t for t in tasks}

        for fut in as_completed(futures):
            task = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                logger.error(f"Task {task} raised: {e}")
                rows = []

            all_rows.extend(rows)
            done += 1

            if done % 50 == 0 or done == len(tasks):
                encoder = task[0]
                partial_path = os.path.join(OUT_DIR, f"ablation_partial_{encoder}.csv")
                encoder_rows = [r for r in all_rows if r["encoder"] == encoder]
                if encoder_rows:
                    pd.DataFrame(encoder_rows).to_csv(partial_path, index=False)
                logger.info(f"[{done}/{len(tasks)}] partial saved → {partial_path}")

    final_path = os.path.join(OUT_DIR, "ablation.csv")
    pd.DataFrame(all_rows).to_csv(final_path, index=False)
    logger.info(f"Done. {len(all_rows)} rows → {final_path}")


if __name__ == "__main__":
    main()
