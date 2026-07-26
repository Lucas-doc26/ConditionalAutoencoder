import os
import csv
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score

from fusion import (
    load_npy, load_ground_truth, align_predictions,
    fusion_sum, fusion_mean, fusion_max, fusion_mult, fusion_vote,
    save_row,
)


def run_combination_fusions():

    base_pattern = (
        "/home/lucas/representation-fusion/Modelos/Modelo_Kyoto-{encoder}/"
        "Classificador-{dataset_encoder}/Resultados/Treinados_em_{dataset_train}/"
        "{dataset_test}/batches-{batch}.npy"
    )

    encoder_to_trains = {
        "CNR":   ["PUC", "UFPR04", "UFPR05"],
        "PKLot": [
            "camera1", "camera2", "camera3", "camera4", "camera5",
            "camera6", "camera7", "camera8", "camera9",
        ],
    }
    encoder_to_tests = {
        "CNR": [
            "camera1", "camera2", "camera3", "camera4", "camera5",
            "camera6", "camera7", "camera8", "camera9",
            "PUC", "UFPR04", "UFPR05",
        ],
        "PKLot": [
            "camera1", "camera2", "camera3", "camera4", "camera5",
            "camera6", "camera7", "camera8", "camera9",
        ],
    }
    batches = ["64", "128", "256", "512", "1024"]

    fusion_methods = {
        "sum": fusion_sum,
        "mean_probs": fusion_mean,
        "max": fusion_max,
        "mult": fusion_mult,
        "vote": fusion_vote,
    }

    csv_results_path = "fusion_combinations.csv"

    for dataset_encoder, datasets_train in encoder_to_trains.items():
        for dataset_train in datasets_train:
            for batch in batches:
                for dataset_test in encoder_to_tests[dataset_encoder]:

                    print(f"\nProcessing {dataset_train} | {dataset_encoder} | {dataset_test} | batch {batch}")

                    models_outputs = []

                    for encoder in range(10):
                        path = base_pattern.format(
                            encoder=encoder,
                            dataset_encoder=dataset_encoder,
                            dataset_train=dataset_train,
                            dataset_test=dataset_test,
                            batch=batch,
                        )
                        if os.path.exists(path):
                            probs, ids = load_npy(path)
                            models_outputs.append((encoder, probs, ids))

                    if len(models_outputs) < 2:
                        print("Less than 2 models found. Skipping.")
                        continue

                    gt_ids, gt_labels = load_ground_truth(dataset_test)

                    plain_outputs = [(p, ids) for _, p, ids in models_outputs]
                    stacked_all, y_true, _ = align_predictions(plain_outputs, gt_ids, gt_labels)
                    encoder_indices = [enc for enc, _, _ in models_outputs]

                    n_available = len(models_outputs)

                    for n_models in range(2, n_available + 1):
                        for combo in combinations(range(n_available), n_models):
                            stacked_sub = stacked_all[list(combo)]
                            combo_encoders = tuple(encoder_indices[i] for i in combo)

                            for name, func in fusion_methods.items():
                                fused_probs = func(stacked_sub)
                                preds = np.argmax(fused_probs, axis=1)

                                acc  = accuracy_score(y_true, preds)
                                prec = precision_score(y_true, preds, average="macro", zero_division=0)
                                f1   = f1_score(y_true, preds, average="macro", zero_division=0)
                                rec  = recall_score(y_true, preds, average="macro", zero_division=0)
                                try:
                                    auc = roc_auc_score(y_true, fused_probs[:, 1])
                                except ValueError:
                                    auc = float("nan")

                                row = {
                                    "tipo_encoder": "Modelo_Kyoto",
                                    "dataset_encoder": dataset_encoder,
                                    "dataset_train": dataset_train,
                                    "dataset_test": dataset_test,
                                    "batch": batch,
                                    "n_models": n_models,
                                    "combination": str(combo_encoders),
                                    "tecnica_fusao": name,
                                    "precision": round(prec, 6),
                                    "accuracy": round(acc, 6),
                                    "f1": round(f1, 6),
                                    "recall": round(rec, 6),
                                    "auc": round(auc, 6) if not np.isnan(auc) else "nan",
                                }

                                save_row(csv_results_path, row)

                    print(f"Saved {n_available} models, combinations 2..{n_available}.")


if __name__ == "__main__":
    run_combination_fusions()
