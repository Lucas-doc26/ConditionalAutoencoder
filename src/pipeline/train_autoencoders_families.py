"""Pré-treino dos autoencoders de todas as famílias, espelhando Modelos/.

Para cada família (ae, skip, vae, joint, jointskip) × índice × base
(CNR, PKLot, Kyoto), treina o autoencoder em 64×64 e salva os pesos em
Modelos/{prefix}-{i}/Modelo-Base/Pesos/{prefix}-{i}_Base-{base}[_encoder|_decoder].pth
— a mesma estrutura dos modelos legados Modelo_Kyoto-*, mas em PyTorch.

Idempotente: pula jobs cujos três .pth já existem (use --no-resume para retreinar).
MLflow: sem --mlflow, loga no file store local ./mlruns (não precisa de servidor).

Uso:
    python -m src.pipeline.train_autoencoders_families -e 10 --families all --bases CNR,PKLot,Kyoto
"""

import argparse
import os

import mlflow
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, split_jobs
from src.config.families import FAMILIES, BASES, resolve_families, resolve_indices
from src.config.paths import CSV_DIR
from src.models.variational_autoencoder import vae_loss
from src.utils.datasets import CustomImageDataset
from src.utils.model_paths import base_weights_path, model_dir
from src.utils.plot import plot_reconstruction
from src.utils.transform import return_transform_64

NUM_WORKS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True


def unique_devices():
    return list(dict.fromkeys(Config().DEVICES))


def dataset_available(base):
    """Confere se as imagens da base existem (amostra a primeira linha do CSV)."""
    csv_path = CSV_DIR / base / f"{base}_autoencoder_train.csv"
    if not csv_path.is_file():
        return False, f"CSV inexistente: {csv_path}"
    with open(csv_path) as f:
        f.readline()
        first = f.readline().strip().split(",")[0]
    if not first or not os.path.exists(first):
        return False, f"imagens indisponíveis (ex.: {first})"
    return True, ""


def job_done(family, index, base):
    return all(
        base_weights_path(family.prefix, index, base, part).is_file()
        for part in (None, "encoder", "decoder")
    )


def make_loader(base, split, batch_size, shuffle):
    dataset = CustomImageDataset(
        str(CSV_DIR / base / f"{base}_autoencoder_{split}.csv"),
        autoencoder=True,
        transform=return_transform_64(),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKS,
        pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS,
    )


def train_one(device, family, index, base, num_epochs=10, batch_size=32, lr=1e-3):
    is_vae = family.key == "vae"
    model_class = family.autoencoders[index]
    model_name = f"{family.prefix}-{index}"

    train_loader = make_loader(base, "train", batch_size, shuffle=True)
    val_loader = make_loader(base, "validation", batch_size, shuffle=False)
    test_loader = make_loader(base, "test", batch_size, shuffle=False)

    mlflow.set_experiment(f"AE64_{family.prefix}")

    with mlflow.start_run(run_name=f"{model_name}_Base-{base}"):
        model = model_class().to(device)
        criterion = vae_loss if is_vae else nn.MSELoss()

        mlflow.log_params({
            "model": model_name,
            "family": family.key,
            "dataset": base,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "input_shape": "3x64x64",
            "latent_dim": model.latent_dim,
            "loss": "VAE Loss (MSE + KL)" if is_vae else "MSE",
        })

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"{model_name} Base-{base} [{epoch+1}/{num_epochs}]", leave=False)
            for x, y, _ in pbar:
                x, y = x.to(device), y.to(device)

                if is_vae:
                    x_hat, mu, logvar = model(x)
                    loss, _, _ = vae_loss(x_hat, x, mu, logvar)
                else:
                    out = model(x)
                    loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y, _ in val_loader:
                    x, y = x.to(device), y.to(device)
                    if is_vae:
                        x_hat, mu, logvar = model(x)
                        loss, _, _ = vae_loss(x_hat, x, mu, logvar)
                    else:
                        out = model(x)
                        loss = criterion(out, y)
                    val_loss += loss.item()

            val_loss /= max(len(val_loader), 1)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            print(f"{model_name} Base-{base} epoch {epoch+1}/{num_epochs} "
                  f"train={train_loss:.5f} val={val_loss:.5f}")

        # Plot de reconstrução em Modelos/{prefix}-{i}/Plots/
        model.eval()
        with torch.no_grad():
            for x, _, _ in test_loader:
                x = x[:8].to(device)
                recon = model(x)
                break
        if is_vae:
            recon = recon[0]
        plots_dir = model_dir(family.prefix, index) / "Plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_reconstruction(x, recon, model_name, f"Base-{base}", save_path=str(plots_dir) + "/")

        weights = base_weights_path(family.prefix, index, base)
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights)
        torch.save(model.encoder.state_dict(), base_weights_path(family.prefix, index, base, "encoder"))
        torch.save(model.decoder.state_dict(), base_weights_path(family.prefix, index, base, "decoder"))
        print(f"✓ {model_name} Base-{base} salvo em {weights.parent}")


def worker(rank, jobs_split, devices, epochs, use_mlflow):
    device = devices[rank % len(devices)]
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    if use_mlflow:
        mlflow.set_tracking_uri(Config().IP_LOCAL)

    for family_key, index, base in jobs_split[rank]:
        family = FAMILIES[family_key]
        train_one(device, family, index, base, num_epochs=epochs)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def build_jobs(args):
    jobs, skipped = [], []
    bases = [b.strip() for b in args.bases.split(",")]
    for base in bases:
        if base not in BASES:
            raise ValueError(f"Base inválida: {base}. Use {BASES}")
        ok, reason = dataset_available(base)
        if not ok:
            print(f"[pulando base {base}] {reason}")
            continue
        for family in resolve_families(args.families):
            for index in resolve_indices(args.indices, family):
                if not args.no_resume and job_done(family, index, base):
                    skipped.append((family.prefix, index, base))
                    continue
                jobs.append((family.key, index, base))
    if skipped:
        print(f"{len(skipped)} jobs já concluídos (resume) — use --no-resume para retreinar")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("--families", type=str, default="all",
                        help="all ou lista: ae,skip,vae,joint,jointskip")
    parser.add_argument("--bases", type=str, default=",".join(BASES))
    parser.add_argument("--indices", type=str, default="all", help="all | 0-9 | 0,3,7")
    parser.add_argument("--no-resume", action="store_true",
                        help="Retreina mesmo que os .pth já existam")
    parser.add_argument("--mlflow", action="store_true",
                        help="Loga no servidor MLflow (Config.IP_LOCAL); default: ./mlruns local")
    args = parser.parse_args()

    jobs = build_jobs(args)
    print(f"Total de jobs de autoencoder: {len(jobs)}")
    if not jobs:
        return

    devices = unique_devices()
    n_procs = min(len(devices), len(jobs))

    if n_procs <= 1 or devices[0] == "cpu":
        worker(0, [jobs], devices, args.epochs, args.mlflow)
        return

    jobs_split = split_jobs(jobs, n_procs)
    print(f"Processos: {n_procs} | Jobs por processo: {[len(j) for j in jobs_split]}")
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(jobs_split, devices, args.epochs, args.mlflow), nprocs=n_procs, join=True)


if __name__ == "__main__":
    main()
