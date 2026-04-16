import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

import mlflow

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from src.utils.datasets import CustomImageDataset
from src.models.joint_autoencoder import (
    JointAutoencoder0, JointAutoencoder1,
    JointSkipAutoencoder0, JointSkipAutoencoder1
)
from src.config.config import Config
from src.utils.plot import plot_reconstruction, denormalize
from src.utils.image_metrics import calculate_all_metrics_torch
from src.models.loss.autoencoder_loss import orthogonal_loss

# ─────────────────────────────────────────────
# Helpers de forward para lidar com skip models
# ─────────────────────────────────────────────

def forward_model(model, x):
    """
    Executa encoder → decoder corretamente tanto para modelos normais
    quanto para skip autoencoders (que retornam tuplas com skip connections).
    
    Retorna:
        recon  : imagem reconstruída  (B, C, H, W)
        z      : vetor latente puro   (B, latent_dim)
    """
    z_out = model.encoder(x)
    if isinstance(z_out, (tuple, list)):
        # Skip autoencoder: z_out = (z, x1, x2, ...) 
        recon = model.decoder(*z_out)
        z = z_out[0]          # apenas o vetor latente, sem as skip features
    else:
        recon = model.decoder(z_out)
        z = z_out
    return recon, z


# ─────────────────────────────────────────────
# Loss combinada — 100% PyTorch, compatível com .backward()
# ─────────────────────────────────────────────

def combined_loss(recons: list, embeddings: list, target: torch.Tensor):
    """
    Loss no intervalo [0, 1]:
      - MSE média entre todos os modelos  (peso 0.3)
      - Perda de ortogonalidade entre embeddings (peso 0.7)
    
    Args:
        recons     : lista de tensores reconstruídos [recon0, recon1, ...]
        embeddings : lista de vetores latentes z  (já extraídos, sem skip features)
        target     : imagem alvo
    """
    mse_losses = torch.stack([F.mse_loss(r, target) for r in recons])
    mse_loss = mse_losses.mean()

    # orthogonal_loss espera lista de tensores z
    ort_loss = orthogonal_loss(embeddings).clamp(0.0, 1.0)

    return 0.3 * mse_loss + 0.7 * ort_loss


# ─────────────────────────────────────────────
# Plot de clusters — 4 encoders
# ─────────────────────────────────────────────

MODEL_NAMES  = ["JointAE-0", "JointAE-1", "JointSkipAE-0", "JointSkipAE-1"]
MARKERS      = ["o", "x", "s", "^"]
CMAPS        = ["tab10", "Set1", "tab20b", "Set2"]

def plot_cluster(representations: list, y_true: list, save_path: str):
    """
    Gera scatter plot PCA para N encoders lado a lado num único gráfico.

    Args:
        representations : lista de N listas de tensores (um por modelo)
        y_true          : labels verdadeiras (flat list ou array)
        save_path       : onde salvar a figura
    
    Returns:
        mean_distance   : distância euclidiana média entre AE-0 e AE-1 (2D)
    """
    # Concatena todos os batches de cada modelo
    reps_np = [torch.cat(r, dim=0).numpy() for r in representations]
    n = reps_np[0].shape[0]
    y_arr = np.array(y_true[:n])

    # PCA conjunta para projeção comparável
    rep_concat = np.concatenate(reps_np, axis=0)          # (N*n, latent_dim)
    pca = PCA(n_components=2)
    rep_2d_all = pca.fit_transform(rep_concat)             # (N*n, 2)

    reps_2d = [rep_2d_all[i * n : (i + 1) * n] for i in range(len(reps_np))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (rep2d, name, marker, cmap) in enumerate(
        zip(reps_2d, MODEL_NAMES, MARKERS, CMAPS)
    ):
        sc = axes[i].scatter(
            rep2d[:, 0], rep2d[:, 1],
            c=y_arr, cmap=cmap, marker=marker,
            alpha=0.75, s=20
        )
        axes[i].set_title(name, fontsize=13)
        axes[i].set_xlabel("PC1")
        axes[i].set_ylabel("PC2")
        plt.colorbar(sc, ax=axes[i], label="Classe")

    plt.suptitle("PCA — Representações dos 4 Encoders", fontsize=15, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Distância média entre os dois primeiros encoders como proxy de diversidade
    mean_distance = euclidean_distances(reps_2d[0], reps_2d[1]).mean()
    return float(mean_distance)


# ─────────────────────────────────────────────
# Plot de reconstruções — 4 modelos numa figura
# ─────────────────────────────────────────────

def plot_all_reconstructions(originals, recons_list, save_path, n_images=8):
    """
    Gera grid: linhas = [Original, Model0, Model1, SkipModel0, SkipModel1]
               colunas = amostras do batch

    Args:
        originals    : tensor (B, C, H, W) — imagens originais
        recons_list  : lista de 4 tensores reconstruídos
        save_path    : caminho para salvar a figura
        n_images     : quantas amostras exibir
    """
    row_labels = ["Original"] + MODEL_NAMES
    n_rows = len(row_labels)
    n_col  = min(n_images, originals.shape[0])

    fig, axes = plt.subplots(n_rows, n_col, figsize=(2.5 * n_col, 2.5 * n_rows))

    def to_np(t):
        img = denormalize(t.detach().cpu())
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
        return img

    all_rows = [originals] + recons_list

    for row_idx, (row_data, label) in enumerate(zip(all_rows, row_labels)):
        for col_idx in range(n_col):
            ax = axes[row_idx, col_idx]
            ax.imshow(to_np(row_data[col_idx]))
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, rotation=90, labelpad=8)

    plt.suptitle("Reconstruções — 4 Modelos", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ─────────────────────────────────────────────
# Datasets e Loaders
# ─────────────────────────────────────────────

train_ds = CustomImageDataset(
    csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/UFPR04/batches/batch-1024.csv",
    autoencoder=True
)
valid_ds = CustomImageDataset(
    csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/UFPR04/UFPR04_validation.csv",
    autoencoder=True
)
test_ds = CustomImageDataset(
    csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/UFPR04/UFPR04_test.csv",
    autoencoder=False,
    data_per_class=200
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

config = Config()
device = config.DEVICE0


# ─────────────────────────────────────────────
# Treinamento
# ─────────────────────────────────────────────

def train_models(num_epochs=50):
    mlflow.set_experiment("JointAutoencoder-4-Modelos")

    with mlflow.start_run(run_name="4-autoencoders-joint"):

        # ── Modelos ──────────────────────────────────
        joint_0      = JointAutoencoder0().to(device)
        joint_1      = JointAutoencoder1().to(device)
        joint_skip_0 = JointSkipAutoencoder0().to(device)
        joint_skip_1 = JointSkipAutoencoder1().to(device)

        models = [joint_0, joint_1, joint_skip_0, joint_skip_1]

        parameters = []
        for m in models:
            parameters += list(m.parameters())

        # ── Otimizador + Scheduler ────────────────────
        optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        best_val_loss = float("inf")

        epoch_pbar = tqdm(range(num_epochs), desc="Epochs", leave=True)
        for epoch in epoch_pbar:

            # ── Train ─────────────────────────────────
            for m in models:
                m.train()

            running_loss = 0.0

            train_pbar = tqdm(
                train_loader,
                desc=f"Train {epoch + 1}/{num_epochs}",
                leave=False
            )
            for x, y, _ in train_pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    recons, zs = [], []
                    for m in models:
                        recon, z = forward_model(m, x)
                        recons.append(recon)
                        zs.append(z)

                    loss = combined_loss(recons, zs, x)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # ── Validation ────────────────────────────
            for m in models:
                m.eval()

            running_val_loss = 0.0

            with torch.no_grad():
                valid_pbar = tqdm(
                    valid_loader,
                    desc=f"Valid {epoch + 1}/{num_epochs}",
                    leave=False
                )
                for x, y, _ in valid_pbar:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        recons, zs = [], []
                        for m in models:
                            recon, z = forward_model(m, x)
                            recons.append(recon)
                            zs.append(z)

                        val_loss = combined_loss(recons, zs, x)

                    running_val_loss += val_loss.item()
                    valid_pbar.set_postfix(val_loss=f"{val_loss.item():.4f}")

            avg_val_loss = running_val_loss / len(valid_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            scheduler.step()

            epoch_pbar.set_postfix(
                train=f"{avg_train_loss:.4f}",
                valid=f"{avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        # ── Test — métricas de imagem ─────────────────
        for m in models:
            m.eval()

        metrics_sum = {
            name: {k: 0.0 for k in ["MSE", "SSIM", "PSNR", "NCC", "VIF", "SCC"]}
            for name in MODEL_NAMES
        }

        last_x = last_recons = None   # para salvar o plot do último batch

        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Test metrics", leave=False)
            for x, _, _ in test_pbar:
                x = x.to(device, non_blocking=True)

                recons = []
                for m in models:
                    recon, _ = forward_model(m, x)
                    recons.append(recon)

                x_dn     = torch.clamp(denormalize(x.cpu()), 0, 1)
                recons_dn = [torch.clamp(denormalize(r.cpu()), 0, 1) for r in recons]

                for name, r_dn in zip(MODEL_NAMES, recons_dn):
                    batch_m = calculate_all_metrics_torch(x_dn, r_dn)
                    for k in metrics_sum[name]:
                        metrics_sum[name][k] += batch_m[k]

                last_x      = x
                last_recons = recons

        n_test_batches = len(test_loader)
        for name in MODEL_NAMES:
            for k in metrics_sum[name]:
                avg = metrics_sum[name][k] / n_test_batches
                # ex: "test_JointAE-0_mse"
                mlflow.log_metric(f"test_{name}_{k.lower()}", avg)

        # ── Plot: reconstruções dos 4 modelos ────────
        os.makedirs("./results/Joint-4-Models", exist_ok=True)
        recon_save_path = "./results/Joint-4-Models/reconstructions.png"
        plot_all_reconstructions(
            last_x,
            last_recons,
            save_path=recon_save_path,
            n_images=8
        )
        mlflow.log_artifact(recon_save_path, artifact_path="reconstructions")

        # ── Clusters dos 4 encoders ──────────────────
        representations = [[] for _ in models]
        y_true = []

        with torch.no_grad():
            for img, label, _ in tqdm(test_loader, desc="Cluster embeddings"):
                img = img.to(device, non_blocking=True)

                for i, m in enumerate(models):
                    _, z = forward_model(m, img)
                    representations[i].append(z.cpu())

                y_true.extend(label.numpy().tolist())

        cluster_path = "./results/Joint-4-Models/cluster_pca.png"
        mean_distance = plot_cluster(representations, y_true, cluster_path)
        mlflow.log_artifact(cluster_path, artifact_path="cluster")
        mlflow.log_metric("mean_distance_ae0_ae1", mean_distance)
        mlflow.log_metric("best_val_loss", best_val_loss)

        print(f"\n✅ Treinamento concluído.")
        print(f"   Best val loss  : {best_val_loss:.4f}")
        print(f"   Mean dist AE0↔AE1 (PCA): {mean_distance:.4f}")

        return best_val_loss, mean_distance


if __name__ == "__main__":
    train_models(num_epochs=50)