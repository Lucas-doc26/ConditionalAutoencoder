import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────── fusion methods ───────────────────────────────────

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
    preds_models = np.argmax(stacked, axis=2)  # [M, N]
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
    "sum": fusion_sum,
    "mean_probs": fusion_mean,
    "max": fusion_max,
    "mult": fusion_mult,
    "vote": fusion_vote,
}

# ─────────────────────────── constants ────────────────────────────────────────

N_MODELS_TOTAL = 10
ALL_MODEL_IDS   = list(range(N_MODELS_TOTAL))

DICT_COMBINATIONS = {
    "PKLot": ['camera1', 'camera2', 'camera3', 'camera4', 'camera5',
              'camera6', 'camera7', 'camera8', 'camera9'],
    "CNR":   ['UFPR04', 'UFPR05', 'PUC'],
    "Kyoto": ['UFPR04', 'UFPR05', 'PUC',
              'camera1', 'camera2', 'camera3', 'camera4', 'camera5',
              'camera6', 'camera7', 'camera8', 'camera9'],
}

TESTES  = ['UFPR04', 'UFPR05', 'PUC',
           'camera1', 'camera2', 'camera3', 'camera4', 'camera5',
           'camera6', 'camera7', 'camera8', 'camera9']
BATCHES = [64, 128, 256, 512, 1024]

BASE_MODEL_DIR = "/home/lucas.ocunha/Downloads/PIBIC-ANO-PASSADO/Modelos"
BASE_CSV_DIR   = "/home/lucas.ocunha/Downloads/PIBIC-ANO-PASSADO/CSV"
OUT_DIR        = "/home/lucas.ocunha/ConditionalAutoencoder/resultados_pibic_ano_passado"

# ─────────────────────────── helpers ──────────────────────────────────────────

def npy_path(encoder, classificador, teste, batch, model_id):
    return (
        f"{BASE_MODEL_DIR}/Modelo_Kyoto-{model_id}/"
        f"Classificador-{encoder}/Resultados/"
        f"Treinados_em_{classificador}/{teste}/batches-{batch}.npy"
    )


def check_and_load_arrays(encoder, classificador, teste, batch):
    """
    Loads all 10 model arrays for a given task.
    Returns (arrays_dict, n_samples) or raises RuntimeError with a reason.
    """
    arrays = {}
    n_samples_set = set()

    for mid in ALL_MODEL_IDS:
        p = npy_path(encoder, classificador, teste, batch, mid)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing: {p}")
        arr = np.load(p)          # load fully once; faster than mmap for repeated access
        arrays[mid] = arr.astype(np.float32, copy=False)
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
                return y
    raise ValueError(f"No label CSV with {n_samples} rows for '{teste}'")


def sample_half(all_combs):
    """Return ~50% of combinations, drawn randomly without replacement."""
    k = max(1, len(all_combs) // 2)
    return random.sample(all_combs, k)


# ─────────────────────────── core worker ──────────────────────────────────────

def process_task(args):
    """
    Worker function executed in a separate process.
    Loads arrays once, iterates over n_models=2..10 with 50% sampled combos.
    Returns a list of row-dicts.
    """
    encoder, classificador, teste, batch = args
    rows = []

    # ── check directory exists ──
    ref_dir = (
        f"{BASE_MODEL_DIR}/Modelo_Kyoto-0/"
        f"Classificador-{encoder}/Resultados/"
        f"Treinados_em_{classificador}/{teste}"
    )
    if not os.path.isdir(ref_dir):
        return rows

    # ── load arrays (once per task) ──
    try:
        arrays, n_samples = check_and_load_arrays(encoder, classificador, teste, batch)
    except Exception as e:
        logger.warning(f"[skip load] {encoder}/{classificador}/{teste}/batch={batch}: {e}")
        return rows

    # ── load ground truth ──
    try:
        y_true = load_y_true(teste, n_samples)
    except Exception as e:
        logger.warning(f"[skip y_true] {encoder}/{classificador}/{teste}/batch={batch}: {e}")
        return rows

    # ── iterate over ensemble sizes ──
    for n in range(2, N_MODELS_TOTAL + 1):
        all_combs = list(combinations(ALL_MODEL_IDS, n))
        selected  = sample_half(all_combs)

        for comb in selected:
            stacked = np.stack([arrays[mid] for mid in comb], axis=0)  # [n, N, C]
            for fname, ffunc in FUSION_METHODS.items():
                y_pred = np.argmax(ffunc(stacked), axis=1)
                acc    = accuracy_score(y_true, y_pred)
                rows.append({
                    "encoder":       encoder,
                    "classificador": classificador,
                    "teste":         teste,
                    "batch":         batch,
                    "n_models":      n,
                    "combination":   comb,
                    "fusion_method": fname,
                    "value":         acc,
                })

    return rows


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build task list
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
    lock     = threading.Lock()   # used only in the main thread for safe accumulation
    done     = 0

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

            # ── partial save every 20 completed tasks ──
            if done % 20 == 0 or done == len(tasks):
                encoder = task[0]
                partial_path = os.path.join(OUT_DIR, f"combinacoes_intermediaria_{encoder}.csv")
                encoder_rows = [r for r in all_rows if r["encoder"] == encoder]
                if encoder_rows:
                    pd.DataFrame(encoder_rows).to_csv(partial_path, index=False)
                logger.info(f"[{done}/{len(tasks)}] saved partial → {partial_path}")

    # ── final save ──
    final_path = os.path.join(OUT_DIR, "combinacoes.csv")
    pd.DataFrame(all_rows).to_csv(final_path, index=False)
    logger.info(f"Done. Final CSV → {final_path}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()