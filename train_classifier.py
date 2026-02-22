import os

import matplotlib
matplotlib.use("Agg")


import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score
from mlflow.tracking import MlflowClient

from src.config import *
from src.utils.datasets import CustomImageDataset
from src.utils.transform import return_transform
from src.models import *
from src.utils.plot import log_confusion_matrix_mlflow

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Utils
def make_test_loader(csv_path, transform, batch_size, pin_memory):
    dataset = CustomImageDataset(
        csv_path,
        transform=transform,
        autoencoder=False
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        prefetch_factor=None,
        persistent_workers=False
    )


# Train + Test
def train_classifier(
    gpu_id,
    model_encoder,
    dataset_encoder_name,
    dataset_classifier_name,
    batch_size=32,
    num_epochs=10,
    lr=1e-3
):
    device = Config.DEVICES[gpu_id]
    transform = return_transform()
    pin_memory = isinstance(device, str) and device.startswith("cuda")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_root = os.path.join(base_dir, "CSV")

    batch_sizes_csv = [64, 128, 256, 512, 1024]
    datasets_test = [
        "PUC", "UFPR04", "UFPR05",
        "camera1", "camera2", "camera3",
        "camera4", "camera5", "camera6",
        "camera7", "camera8", "camera9"
    ]

    # -----------------------------
    # MLflow experiment
    # -----------------------------
    experiment = (
        f"Classifier_Skip_{model_encoder.__name__[-1]}"
        if model_encoder.__name__.startswith("Skip")
        else f"Classifier_AE_{model_encoder.__name__[-1]}"
    )

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)

    mlflow.set_experiment(experiment)

    # -----------------------------
    # Load encoder ONCE
    # -----------------------------
    encoder = model_encoder().to(device)

    if re.search(r"^Skip", model_encoder.__name__):
        name_encoder = f"SkipAutoencoder{model_encoder.__name__[-1]}"
    else:
        name_encoder = f"Autoencoder{model_encoder.__name__[-1]}"

    encoder.load_state_dict(
        torch.load(
            f"models/{name_encoder}_{dataset_encoder_name}/encoder.pth",
            map_location=device
        )
    )

    for p in encoder.parameters():
        p.requires_grad = False

    latent_dim = encoder.latent_dim
    for batch_size_csv in batch_sizes_csv:
        
        model = Classifier(encoder, latent_dim=latent_dim, num_classes=2).to(device)
        
        train_csv = os.path.join(
            csv_root,
            dataset_classifier_name,
            "batches",
            f"batch-{batch_size_csv}.csv"
        )
        train_dataset = CustomImageDataset(
            train_csv,
            transform=transform,
            autoencoder=False
        )
        train_dataset_mlflow = mlflow.data.from_pandas(
            pd.read_csv(train_csv), source=train_csv, name=f"{dataset_classifier_name}_{batch_size_csv}"
        )
        
        val_dataset = CustomImageDataset(
            os.path.join(
                csv_root,
                dataset_classifier_name,
                f"{dataset_classifier_name}_validation.csv"
            ),
            transform=transform,
            autoencoder=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=False
        )

        model_name = f"{model_encoder.__name__}_Classifier"
        
        # -----------------------------
        # Cache test loaders
        # -----------------------------
        test_loaders = {
            name: make_test_loader(
                os.path.join(csv_root, name, f"{name}_test.csv"),
                transform,
                batch_size,
                pin_memory
            )
            for name in datasets_test
        }

        # ========================================================
        # MLflow run
        # ========================================================
        with mlflow.start_run(
            run_name=f"{model_name}_{dataset_classifier_name}_{batch_size_csv}"
        ):
            mlflow.log_param("model", model_name)
            mlflow.log_param("encoder_dataset", dataset_encoder_name)
            mlflow.log_param("dataset_classifier", dataset_classifier_name)
            mlflow.log_param("epochs", num_epochs)
            mlflow.log_param("n_images_to_train", batch_size_csv)
            mlflow.log_param("lr", lr)
            mlflow.log_param("loss", "CrossEntropy")
            mlflow.log_param("input_shape", "3x128x128")
            mlflow.log_input(train_dataset_mlflow, context="training")


            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Train
            for epoch in tqdm(
                range(num_epochs),
                desc=f"[GPU {gpu_id}] Epochs",
                position=gpu_id
            ):
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0

                for x, y, _ in train_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    out = model(x)
                    loss = criterion(out, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_correct += (out.argmax(1) == y).sum().item()
                    train_total += y.size(0)

                train_loss /= len(train_loader)
                train_acc = train_correct / train_total

                # -------------------------
                # Validation
                # -------------------------
                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0

                with torch.no_grad():
                    for x, y, _ in val_loader:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)

                        out = model(x)
                        loss = criterion(out, y)

                        val_loss += loss.item()
                        val_correct += (out.argmax(1) == y).sum().item()
                        val_total += y.size(0)

                val_loss /= len(val_loader)
                val_acc = val_correct / val_total

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)

            # TEST (offline style)
            test_results = {}

            for name, loader in test_loaders.items():
                y_true, y_pred = [], []
                all_logits, all_ids = [], []

                model.eval()
                with torch.no_grad():
                    for x, y, idx in tqdm(loader,  desc=f"[GPU {gpu_id}] Testing on {name}", position=gpu_id):
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)

                        out = model(x)
                        preds = out.argmax(1)   

                        y_true.extend(y.cpu().numpy())
                        y_pred.extend(preds.cpu().numpy())
                        probs = torch.softmax(out, dim=1)
                        all_logits.append(probs.cpu().numpy().astype(np.float16))
                        all_ids.extend(idx.cpu().numpy())
                

                test_results[name] = {
                    "acc": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred),
                    "f1": f1_score(y_true, y_pred),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "logits": np.concatenate(all_logits),
                    "ids": np.array(all_ids),
                }

                del y_true, y_pred, all_logits, all_ids
                torch.cuda.empty_cache()
                
            # Log results
            for name, r in test_results.items():
                mlflow.log_metric(f"test_acc-{name}", r["acc"])
                mlflow.log_metric(f"test_precision-{name}", r["precision"])
                mlflow.log_metric(f"test_f1-{name}", r["f1"])

                log_confusion_matrix_mlflow(
                    r["y_true"],
                    r["y_pred"],
                    artifact_name=f"confusion_matrix/{name}/{batch_size_csv}.png",
                    normalize=False
                )

                save_dir = (
                    f"models/{model_name}_{dataset_classifier_name}/encoder_{dataset_encoder_name}/"
                    f"{batch_size_csv}/preds/{name}"
                )
                os.makedirs(save_dir, exist_ok=True)
                np.savez_compressed(
                        f"{save_dir}/outputs.npz",
                        probs=r["logits"],          # já float16
                        ids=r["ids"]
                    )

            #mlflow.pytorch.log_model(model, "classifier")

            """os.makedirs(
                f"models/{model_name}_{dataset_classifier_name}/"
                f"{batch_size_csv}/weights/",
                exist_ok=True
            )"""
            """torch.save(
                model.state_dict(),
                f"models/{model_name}_{dataset_classifier_name}/"
                f"{batch_size_csv}/weights/classifier.pth"
            )"""

            mlflow.end_run()


def worker(rank, jobs_by_gpu):
    gpu = Config.DEVICES[rank]
    torch.cuda.set_device(gpu)

    mlflow.set_tracking_uri(Config.IP_LOCAL)

    my_jobs = jobs_by_gpu[rank]

    print(
        f"[Rank {rank}] GPU {gpu} | "
        f"Jobs: {len(my_jobs)}"
    )

    for model, enc_ds, cls_ds, epochs in my_jobs:
        train_classifier(
            gpu_id=rank,
            model_encoder=model,
            dataset_encoder_name=enc_ds,
            dataset_classifier_name=cls_ds,
            batch_size=32,
            num_epochs=epochs
        )

        torch.cuda.empty_cache()

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10)
    args = parser.parse_args()

    ENCODERS = [
        Encoder0, Encoder1, Encoder2, Encoder3, Encoder4,
        Encoder5, Encoder6, Encoder7, Encoder8, Encoder9
    ]

    SKIP_ENCODERS = [
        SkipEncoder0, SkipEncoder1, SkipEncoder2, SkipEncoder3,
        SkipEncoder4, SkipEncoder5, SkipEncoder6, SkipEncoder7,
        SkipEncoder8, SkipEncoder9
    ]

    VAE_ENCODERS = [
        VariationalEncoder0, VariationalEncoder1, VariationalEncoder2, VariationalEncoder3, VariationalEncoder4,
        VariationalEncoder5, VariationalEncoder6, VariationalEncoder7, VariationalEncoder8, VariationalEncoder9
    ]

    datasets_encoder = ["CNR", "PKLot"]
    datasets_classifier = [
        "PUC", "UFPR04", "UFPR05",
        "camera1", "camera2", "camera3",
        "camera4", "camera5", "camera6",
        "camera7", "camera8", "camera9"
    ]

    encoder_jobs = [
        (model, enc_ds, cls_ds, args.epochs)
        for enc_ds in datasets_encoder
        for model in ENCODERS
        for cls_ds in datasets_classifier
    ]

    vae_encoder_jobs = [
        (model, enc_ds, cls_ds, args.epochs)
        for enc_ds in datasets_encoder
        for model in VAE_ENCODERS
        for cls_ds in datasets_classifier
    ]

    skip_encoder_jobs = [
        (model, enc_ds, cls_ds, args.epochs)
        for enc_ds in datasets_encoder
        for model in SKIP_ENCODERS
        for cls_ds in datasets_classifier
    ]
    import math

    half = math.ceil(len(vae_encoder_jobs) / 2)

    vae_gpu0 = vae_encoder_jobs[:half]
    vae_gpu1 = vae_encoder_jobs[half:]

    jobs_by_gpu = {
        0: encoder_jobs + vae_gpu0,
        1: skip_encoder_jobs + vae_gpu1,
    }

    assert len(Config.DEVICES) >= 2, "Você precisa de pelo menos 2 GPUs"

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(jobs_by_gpu,),
        nprocs=2
    )
