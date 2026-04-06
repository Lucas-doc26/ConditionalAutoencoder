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


def ensure_probability_array(arr, path):
    if arr.ndim != 2:
        raise ValueError(f"Unsupported array shape {arr.shape} in {path}; expected [N, C]")
    if arr.shape[0] == 0:
        raise ValueError(f"Empty probability array in {path}")
    if arr.shape[1] < 2:
        raise ValueError(f"Array in {path} must contain at least 2 classes, got shape {arr.shape}")

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"NaN or infinite values found in {path}")

    if not np.all((arr >= 0.0) & (arr <= 1.0)):
        logger.warning(f"Values outside [0, 1] found in {path}; check whether this is logits instead of probabilities")

    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        logger.warning(
            f"Row sums in {path} are not all ~1 (min={row_sums.min():.4f}, max={row_sums.max():.4f}); "
            "this can indicate misformatted probabilities or misaligned labels"
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
        arr = arr.astype(np.float32, copy=False)
        ensure_probability_array(arr, p)
        arrays[mid] = arr
        n_samples_set.add(arr.shape[0])

    if len(n_samples_set) != 1:
        raise ValueError(f"Inconsistent sample counts: {n_samples_set}")

    return arrays, next(iter(n_samples_set))


def load_y_true(teste, n_samples):
    for suffix in ["_test.csv", ".csv"]:
        p = f"{BASE_CSV_DIR}/{teste}/{teste}{suffix}"
        if os.path.isfile(p):
            df = pd.read_csv(p)
            if "class" not in df.columns:
                raise ValueError(f"Label file {p} does not contain a 'class' column")
            y = df["class"].values
            if len(y) != n_samples:
                raise ValueError(f"CSV {p} has {len(y)} rows but expected {n_samples}")
            return y
    raise ValueError(f"No label CSV with {n_samples} rows for '{teste}'")


def sample_half(all_combs):
    """Return ~50% of combinations, drawn randomly without replacement."""
    k = max(1, len(all_combs) // 2)
    return random.sample(all_combs, k)


# ─────────────────────────── core worker ──────────────────────────────────────

def validate_alignment(arr, y_true, encoder, classificador, teste, batch):
    if arr.shape[0] != len(y_true):
        raise ValueError(
            f"Loaded probabilities have {arr.shape[0]} samples but y_true has {len(y_true)} rows "
            f"for {encoder}/{classificador}/{teste}/batch={batch}"
        )

    if arr.ndim != 2:
        raise ValueError(
            f"Expected probability array with ndim=2, got {arr.ndim} for {encoder}/{classificador}/{teste}/batch={batch}"
        )

    y_true_int = y_true.astype(np.int64)
    unique_classes = np.unique(y_true_int)
    if len(unique_classes) == 2 and np.array_equal(unique_classes, [0, 1]):
        preds = np.argmax(arr, axis=1)
        acc = accuracy_score(y_true_int, preds)
        acc_flip = accuracy_score(y_true_int, 1 - preds)
        if acc_flip > acc + 1e-6:
            raise ValueError(
                f"Detected probable binary label inversion for {encoder}/{classificador}/{teste}/batch={batch}: "
                f"accuracy={acc:.4f}, flipped_accuracy={acc_flip:.4f}"
            )


def process_task(args):
    """
    Worker function executed em um processo separado.
    Loads arrays once, iterates over n_models=2..10 with 50% sampled combos.
    Returns uma lista de dicionários.
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

    # ── validate alignment on the first available model
    try:
        representative = next(iter(arrays.values()))
        validate_alignment(representative, y_true, encoder, classificador, teste, batch)
    except Exception as e:
        logger.warning(f"[skip alignment] {encoder}/{classificador}/{teste}/batch={batch}: {e}")
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