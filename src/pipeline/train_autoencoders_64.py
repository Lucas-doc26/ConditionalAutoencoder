import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import mlflow
import mlflow.pytorch
import torch.multiprocessing as mp
from tqdm import tqdm

from src.config import *
from src.utils.datasets import CustomImageDataset
from src.utils.plot import plot_reconstruction, denormalize
from src.utils.transform import return_transform_64
from src.models import *
from src.utils.image_metrics import calculate_all_metrics_torch
from src.models.joint_autoencoder import (
    JointAutoencoder0, JointAutoencoder1,
    JointSkipAutoencoder0, JointSkipAutoencoder1,
)
from src.models.variational_autoencoder import vae_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_WORKS = 8
PERSISTENT_WORKERS = True
PIN_MEMORY = True

VAE_CLASSES = (
    VariationalAutoencoder0, VariationalAutoencoder1, VariationalAutoencoder2,
    VariationalAutoencoder3, VariationalAutoencoder4, VariationalAutoencoder5,
    VariationalAutoencoder6, VariationalAutoencoder7, VariationalAutoencoder8,
    VariationalAutoencoder9,
)


def train_experiment_autoencoder(
    gpu_id=0,
    model_class=None,
    dataset_name=None,
    batch_size=32,
    num_epochs=10,
    lr=1e-3
):
    device = Config().DEVICES[gpu_id]
    torch.cuda.set_device(device)

    transform = return_transform_64()

    train_dataset = CustomImageDataset(
        os.path.join(PROJECT_ROOT, "CSV", dataset_name, f"{dataset_name}_autoencoder_train.csv"),
        autoencoder=True,
        transform=transform
    )
    val_dataset = CustomImageDataset(
        os.path.join(PROJECT_ROOT, "CSV", dataset_name, f"{dataset_name}_autoencoder_validation.csv"),
        autoencoder=True,
        transform=transform
    )
    test_dataset = CustomImageDataset(
        os.path.join(PROJECT_ROOT, "CSV", dataset_name, f"{dataset_name}_autoencoder_test.csv"),
        autoencoder=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    model_name = model_class.__name__
    mlflow.set_experiment(model_name)

    with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):

        # -------------------------
        # Params
        # -------------------------
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("input_shape", "3x64x64")

        model = model_class().to(device)
        if isinstance(model, VAE_CLASSES):
            criterion = vae_loss
            mlflow.log_param("loss", "VAE Loss (BCE + KL)")
        else:
            criterion = nn.MSELoss()
            mlflow.log_param("loss", "MSE")
        mlflow.log_param("latent_dim", model.latent_dim)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # -------------------------
        # Train / Val
        # -------------------------
        for epoch in range(num_epochs):

            pbar = tqdm(
                train_loader,
                desc=f"Epoch [{epoch+1}/{num_epochs}]",
                leave=False
            )

            model.train()
            train_loss = 0.0
            total_recon = 0
            total_kl = 0
            for x, y, _ in pbar:
                x, y = x.to(device), y.to(device)

                if isinstance(model, VAE_CLASSES):
                    x_hat, mu, logvar = model(x)
                    loss, recon, kl = vae_loss(x_hat, x, mu, logvar)
                    total_recon += recon.item()
                    total_kl += kl.item()
                else:
                    out = model(x)
                    loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            if isinstance(model, VAE_CLASSES):
                recon_avg = total_recon / len(train_loader)
                kl_avg = total_kl / len(train_loader)

                mlflow.log_metric("recon", recon_avg, step=epoch)
                mlflow.log_metric("kl", kl_avg, step=epoch)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y, _ in val_loader:
                    x, y = x.to(device), y.to(device)
                    if isinstance(model, VAE_CLASSES):
                        x_hat, mu, logvar = model(x)
                        loss, _, _ = vae_loss(x_hat, x, mu, logvar)
                    else:
                        out = model(x)
                        loss = criterion(out, y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # -------------------------
        # Test + Plot
        # -------------------------
        model.eval()
        with torch.no_grad():
            for x, _, _ in test_loader:
                x = x[:8].to(device)
                recon = model(x)
                break

        if isinstance(model, VAE_CLASSES):
            recon = recon[0]

        plot_path = plot_reconstruction(
            x, recon, model_name, dataset_name
        )
        mlflow.log_artifact(plot_path, artifact_path="reconstructions")

        metrics_sum = {
            "MSE": 0.0,
            "SSIM": 0.0,
            "PSNR": 0.0,
            "NCC": 0.0,
            "VIF": 0.0,
            "SCC": 0.0
        }

        with torch.no_grad():
            for i, (images, _, _) in enumerate(test_loader):

                if i >= 2:
                    break

                images = images.to(device)
                outputs = model(images)

                if isinstance(model, VAE_CLASSES):
                    outputs = outputs[0]
                outputs = outputs.to(device)

                images_cpu = images.detach().cpu()
                outputs_cpu = outputs.detach().cpu()

                images_dn = torch.clamp(denormalize(images_cpu), 0, 1)
                outputs_dn = torch.clamp(denormalize(outputs_cpu), 0, 1)

                batch_metrics = calculate_all_metrics_torch(
                    images_dn, outputs_dn
                )

                for k in metrics_sum:
                    metrics_sum[k] += batch_metrics[k]

        metrics_avg = {
            k: float(np.mean(v)) for k, v in metrics_sum.items()
        }
        for name, value in metrics_avg.items():
            mlflow.log_metric(f"test_{name.lower()}", value)

        # -------------------------
        # Models
        # -------------------------
        mlflow.pytorch.log_model(model, "autoencoder")

        if hasattr(model, "encoder"):
            mlflow.pytorch.log_model(model.encoder, "encoder")
        if hasattr(model, "decoder"):
            mlflow.pytorch.log_model(model.decoder, "decoder")

        print(f"✓ {model_name} | {dataset_name} finalizado")

        os.makedirs(f"models_64/{model_name}/{dataset_name}", exist_ok=True)

        torch.save(model.state_dict(), f"models_64/{model_name}/{dataset_name}/autoencoder.pth")
        torch.save(model.encoder.state_dict(), f"models_64/{model_name}/{dataset_name}/encoder.pth")
        torch.save(model.decoder.state_dict(), f"models_64/{model_name}/{dataset_name}/decoder.pth")


def worker(rank, jobs_split):
    NUM_GPUS = len(Config().DEVICES)
    gpu_id = rank % NUM_GPUS

    torch.cuda.set_device(gpu_id)

    mlflow.set_tracking_uri(Config().IP_LOCAL)

    my_jobs = jobs_split[rank]

    print(f"[Rank {rank}] -> GPU {gpu_id} | {len(my_jobs)} jobs")

    for model, dataset_encoder, epochs in my_jobs:
        train_experiment_autoencoder(
            model_class=model,
            dataset_name=dataset_encoder,
            num_epochs=epochs,
            gpu_id=int(gpu_id)
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Número de épocas para treinamento")
    parser.add_argument("-m", "--models", type=str, default="all", help="all | vae | skip | ae | joint")

    args = parser.parse_args()

    epochs = args.epochs

    if args.models == "all":
        encoders = [
            Autoencoder0, Autoencoder1, Autoencoder2,
            Autoencoder3, Autoencoder4, Autoencoder5,
            Autoencoder6, Autoencoder7, Autoencoder8,
            Autoencoder9,
            SkipAutoencoder0, SkipAutoencoder1, SkipAutoencoder2,
            SkipAutoencoder3, SkipAutoencoder4, SkipAutoencoder5,
            SkipAutoencoder6, SkipAutoencoder7, SkipAutoencoder8,
            SkipAutoencoder9,
            VariationalAutoencoder0, VariationalAutoencoder1, VariationalAutoencoder2,
            VariationalAutoencoder3, VariationalAutoencoder4, VariationalAutoencoder5,
            VariationalAutoencoder6, VariationalAutoencoder7, VariationalAutoencoder8,
            VariationalAutoencoder9,
            JointAutoencoder0, JointAutoencoder1,
            JointSkipAutoencoder0, JointSkipAutoencoder1,
        ]
    elif args.models == "vae":
        encoders = [
            VariationalAutoencoder0, VariationalAutoencoder1, VariationalAutoencoder2,
            VariationalAutoencoder3, VariationalAutoencoder4, VariationalAutoencoder5,
            VariationalAutoencoder6, VariationalAutoencoder7, VariationalAutoencoder8,
            VariationalAutoencoder9,
        ]
    elif args.models == "skip":
        encoders = [
            SkipAutoencoder0, SkipAutoencoder1, SkipAutoencoder2,
            SkipAutoencoder3, SkipAutoencoder4, SkipAutoencoder5,
            SkipAutoencoder6, SkipAutoencoder7, SkipAutoencoder8,
            SkipAutoencoder9,
        ]
    elif args.models == "ae":
        encoders = [
            Autoencoder0, Autoencoder1, Autoencoder2,
            Autoencoder3, Autoencoder4, Autoencoder5,
            Autoencoder6, Autoencoder7, Autoencoder8,
            Autoencoder9,
        ]
    elif args.models == "joint":
        encoders = [
            JointAutoencoder0, JointAutoencoder1,
            JointSkipAutoencoder0, JointSkipAutoencoder1,
        ]
    else:
        raise ValueError(f"Opção inválida: {args.models}. Use all | vae | skip | ae | joint")

    jobs = []

    for dataset_encoder in ["CNR", "PKLot"]:
        for model in encoders:
            jobs.append((model, dataset_encoder, epochs))

    jobs.sort(key=lambda x: x[0].__name__)

    NUM_GPUS = len(Config().DEVICES)
    PROCS_PER_GPU = 1

    n_procs = NUM_GPUS * PROCS_PER_GPU
    n_procs = min(n_procs, len(jobs))

    jobs_split = split_jobs(jobs, n_procs)

    print(f"Total de jobs: {len(jobs)}")
    print(f"Processos: {n_procs}")
    print(f"Jobs por processo: {[len(j) for j in jobs_split]}")

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(jobs_split,),
        nprocs=n_procs,
        join=True
    )
