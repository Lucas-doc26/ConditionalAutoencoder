import os
import csv
import numpy as np
import pandas as pd
from scipy.special import softmax
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score


# =========================================================
# Load model outputs
# =========================================================

def load_npz(path):
    d = np.load(path, allow_pickle=True)

    if 'logits' in d.files:
        probs = softmax(d['logits'].astype(np.float64), axis=1)
    elif 'probs' in d.files:
        probs = d['probs'].astype(np.float32)
    else:
        raise ValueError(f"No logits/probs found in {path}")

    ids = d['ids'].astype(np.int64)

    return probs.astype(np.float32), ids


# =========================================================
# Load ground truth (ID = índice da linha do CSV)
# =========================================================

def load_ground_truth(dataset_test):

    csv_path = f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_test}/{dataset_test}_test.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    ids = df.index.values.astype(np.int64)
    labels = df["class"].values.astype(np.int64)

    return ids, labels


# =========================================================
# Align predictions with ground truth
# =========================================================

def align_predictions(models_outputs, gt_ids, gt_labels):

    id_to_gt = {int(i): l for i, l in zip(gt_ids, gt_labels)}

    common_ids = sorted(
        set.intersection(*[set(ids.tolist()) for _, ids in models_outputs])
        & set(gt_ids.tolist())
    )

    stacked_probs = []

    for probs, ids in models_outputs:
        id2idx = {int(i): idx for idx, i in enumerate(ids.tolist())}
        rows = [id2idx[int(cid)] for cid in common_ids]
        stacked_probs.append(probs[rows])

    stacked = np.stack(stacked_probs, axis=0)
    y_true = np.array([id_to_gt[int(cid)] for cid in common_ids])

    return stacked, y_true, common_ids


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

    probs = np.eye(C)[final_preds]
    return probs.astype(np.float32)


# =========================================================
# Main Runner
# =========================================================

def run_all_fusions():

    base_pattern = (
        "/home/lucas.ocunha/ConditionalAutoencoder/models/"
        "{type_of_encoder}{encoder}_Classifier/{dataset_train}/encoder_{dataset_encoder}/{batch}/preds/"
        "{dataset_test}/outputs.npz"
    )

    type_of_encoders = ["Encoder", "SkipEncoder", "VariationalEncoder"]
    dataset_encoders = ["CNR", 'PKLot']
    datasets_train = ["camera1","camera2","camera3","camera4","camera5",
        "camera6","camera7","camera8","camera9",
        "PUC","UFPR04","UFPR05"]
    
    dataset_tests = [
        "camera1","camera2","camera3","camera4","camera5",
        "camera6","camera7","camera8","camera9",
        "PUC","UFPR04","UFPR05"
    ]
    batches = ["64", "128", "256", "512", "1024"]

    csv_results_path = "fusion_metrics.csv"

    for type_of_encoder in type_of_encoders:
        for dataset_train in datasets_train:
            for dataset_encoder in dataset_encoders:
                for batch in batches:
                    for dataset_test in dataset_tests:

                        print(f"\nProcessing {type_of_encoder} | {dataset_train} | {dataset_encoder} | {dataset_test} | batch {batch}")

                        models_outputs = []

                        for encoder in range(10):
                            path = base_pattern.format(
                                type_of_encoder=type_of_encoder,
                                encoder=encoder,
                                dataset_train=dataset_train,
                                dataset_encoder=dataset_encoder,
                                dataset_test=dataset_test,
                                batch=batch
                            )

                            if os.path.exists(path):
                                probs, ids = load_npz(path)
                                models_outputs.append((probs, ids))

                        if len(models_outputs) == 0:
                            print("No models found. Skipping.")
                            continue

                        gt_ids, gt_labels = load_ground_truth(dataset_test)
                        stacked, y_true, common_ids = align_predictions(models_outputs, gt_ids, gt_labels)

                        # =====================================================
                        # Individual model statistics
                        # =====================================================

                        individual_acc = []
                        individual_prec = []

                        for probs, ids in models_outputs:

                            id_to_gt = {int(i): l for i, l in zip(gt_ids, gt_labels)}
                            id2idx = {int(i): idx for idx, i in enumerate(ids.tolist())}
                            rows = [id2idx[int(cid)] for cid in common_ids]

                            preds = np.argmax(probs[rows], axis=1)
                            y_true_ind = np.array([id_to_gt[int(cid)] for cid in common_ids])

                            acc = accuracy_score(y_true_ind, preds)
                            prec = precision_score(y_true_ind, preds, average="macro", zero_division=0)

                            individual_acc.append(acc)
                            individual_prec.append(prec)

                        mean_ind_acc = np.mean(individual_acc)
                        std_ind_acc = np.std(individual_acc)
                        mean_ind_prec = np.mean(individual_prec)
                        std_ind_prec = np.std(individual_prec)

                        # Save baseline
                        baseline_row = {
                            "tipo_encoder": type_of_encoder,
                            "dataset_encoder": dataset_encoder,
                            "dataset_train": dataset_train,
                            "dataset_test": dataset_test,
                            "batch": batch,
                            "tecnica_fusao": "mean_individual_models",
                            "precision": round(mean_ind_prec, 6),
                            "precision_std": round(std_ind_prec, 6),
                            "accuracy": round(mean_ind_acc, 6),
                            "accuracy_std": round(std_ind_acc, 6)
                        }

                        save_row(csv_results_path, baseline_row)

                        # =====================================================
                        # Fusion methods
                        # =====================================================

                        fusion_methods = {
                            "sum": fusion_sum,
                            "mean_probs": fusion_mean,
                            "max": fusion_max,
                            "mult": fusion_mult,
                            "vote": fusion_vote
                        }

                        for name, func in fusion_methods.items():

                            fused_probs = func(stacked)
                            preds = np.argmax(fused_probs, axis=1)

                            acc = accuracy_score(y_true, preds)
                            prec = precision_score(y_true, preds, average="macro", zero_division=0)

                            row = {
                                "tipo_encoder": type_of_encoder,
                                "dataset_train": dataset_train,
                                "dataset_encoder": dataset_encoder,
                                "dataset_test": dataset_test,
                                "batch": batch,
                                "tecnica_fusao": name,
                                "precision": round(prec, 6),
                                "precision_std": 0.0,
                                "accuracy": round(acc, 6),
                                "accuracy_std": 0.0
                            }

                            save_row(csv_results_path, row)

                        print("Saved metrics.")


# =========================================================
# CSV Writer
# =========================================================

def save_row(csv_path, row):

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =========================================================
if __name__ == "__main__":
    run_all_fusions()
