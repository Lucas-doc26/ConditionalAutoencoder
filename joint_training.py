import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from itertools import product

import mlflow

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from src.utils.datasets import CustomImageDataset
from src.models.joint_autoencoder import JointAutoencoder0, JointAutoencoder1
from src.config.config import Config
from src.utils.plot import plot_reconstruction, denormalize
from src.utils.image_metrics import *
from src.models.loss.autoencoder_loss import ssim_loss, orthogonal_loss

import numpy as np

def normalize_mse(x, max_mse=1.0):
    # MSE ∈ [0, +∞) → invertido (menor é melhor)
    return np.clip(1 - (x / max_mse), 0, 1)

def normalize_ssim(x):
    # SSIM ∈ [-1, 1] → geralmente [0,1]
    return np.clip((x + 1) / 2, 0, 1)

def normalize_psnr(x, min_psnr=0, max_psnr=50):
    # PSNR ∈ [0, ~50+] (depende do caso)
    return np.clip((x - min_psnr) / (max_psnr - min_psnr), 0, 1)

def normalize_ncc(x):
    # NCC ∈ [-1, 1]
    return np.clip((x + 1) / 2, 0, 1)

def normalize_vif(x, max_vif=1.0):
    # VIF ≥ 0, normalmente até ~1
    return np.clip(x / max_vif, 0, 1)

def normalize_scc(x):
    # SCC ∈ [-1, 1]
    return np.clip((x + 1) / 2, 0, 1)


train = CustomImageDataset(
    csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/UFPR04/batches/batch-1024.csv",
    autoencoder=True
)

valid = CustomImageDataset(
    csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/UFPR04/UFPR04_validation.csv",
    autoencoder=True
)

test = CustomImageDataset(
    csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/UFPR04/UFPR04_test.csv",
    autoencoder=False,
    data_per_class=200
)


train_loader = DataLoader(train, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid, batch_size=32, shuffle=False)
test_loader = DataLoader(test, batch_size=32, shuffle=False)

config = Config()
device = config.DEVICE0

criterion = nn.MSELoss()


def plot_cluster(representations0, representations1, y_true, save_path):

    rep0 = torch.cat(representations0).numpy()
    rep1 = torch.cat(representations1).numpy()

    rep_all = np.concatenate([rep0, rep1], axis=0)

    pca = PCA(n_components=2)
    rep_all_2d = pca.fit_transform(rep_all)

    n = rep0.shape[0]
    rep0_2d = rep_all_2d[:n]
    rep1_2d = rep_all_2d[n:]

    y_true = y_true[:n]

    dist_matrix = euclidean_distances(rep0_2d, rep1_2d)
    mean_distance = dist_matrix.mean()

    plt.figure(figsize=(8,8))

    plt.scatter(
        rep0_2d[:,0], rep0_2d[:,1],
        c=y_true,
        cmap="tab10",
        marker="o",
        label="Encoder0",
        alpha=0.7
    )

    plt.scatter(
        rep1_2d[:,0], rep1_2d[:,1],
        c=y_true,
        cmap="Set1",
        marker="x",
        label="Encoder1",
        alpha=0.7
    )

    plt.legend()
    plt.title("PCA comparison of encoder representations")
    plt.savefig(save_path)
    plt.close()

    return mean_distance


def combined_loss(
    recon0,
    recon1,
    z0,
    z1,
    target,
    mse_weight,
    ssim_weight,
    psnr_weight,
    ncc_weight,
    vif_weight,
    scc_weight,
    rec_weight,
):
    """
    Loss no intervalo [0, 1], onde 0 = bom e 1 = ruim.
    - rec_weight controla o trade-off entre reconstrução e ortogonalidade.
    - pesos das métricas de imagem: mse_weight/ssim_weight e psnr/ncc/vif/scc.
    """

    rec_weight = float(rec_weight)
    rec_weight = max(0.0, min(1.0, rec_weight))

    def differentiable_ncc(a, b, eps=1e-8):
        a0 = a - a.mean()
        b0 = b - b.mean()
        num = (a0 * b0).sum()
        denom = torch.sqrt((a0.pow(2).sum() * b0.pow(2).sum()) + eps)
        return num / denom

    # Métricas de imagem (média entre recon0 e recon1)
    metrics0 = calculate_all_metrics_torch(recon0, target)
    metrics1 = calculate_all_metrics_torch(recon1, target)
    

    mse_l = normalize_mse(((metrics0['MSE'] + metrics1['MSE']) / 2))
    ssim_l = normalize_ssim(((metrics0['SSIM'] + metrics1['SSIM']) / 2))
    psnr_l = normalize_psnr(((metrics0['PSNR'] + metrics1['PSNR']) / 2))
    ncc_l = normalize_ncc(((metrics0['NCC'] + metrics1['NCC']) / 2))
    vif_l = normalize_vif(((metrics0['VIF'] + metrics1['VIF']) / 2))
    scc_l = normalize_scc(((metrics0['SCC'] + metrics1['SCC']) / 2))

    # Pesos das métricas (normaliza para soma=1 -> rec_loss em [0,1])
    image_weights = {
        "MSE": float(mse_weight),
        "SSIM": float(ssim_weight),
        "PSNR": float(psnr_weight),
        "NCC": float(ncc_weight),
        "VIF": float(vif_weight),
        "SCC": float(scc_weight),
    }
    w_sum = sum(image_weights.values())
    if w_sum <= 0:
        image_weights = {k: 1.0 / len(image_weights) for k in image_weights}
    else:
        image_weights = {k: v / w_sum for k, v in image_weights.items()}

    rec_loss = (
        image_weights["MSE"] * mse_l
        + image_weights["SSIM"] * ssim_l
        + image_weights["PSNR"] * psnr_l
        + image_weights["NCC"] * ncc_l
        + image_weights["VIF"] * vif_l
        + image_weights["SCC"] * scc_l
    )

    # Orthogonalidade: cosine^2 média => [0,1] (0 bom, 1 ruim)
    ort_loss = orthogonal_loss([z0, z1]).clamp(0.0, 1.0)

    total_loss = rec_weight * rec_loss + (1.0 - rec_weight) * ort_loss
    return total_loss.clamp(0.0, 1.0)



def train_models(
    mse_weight,
    ssim_weight,
    psnr_weight,
    ncc_weight,
    vif_weight,
    scc_weight,
    rec_weight,
    num_epochs=50,
):

    
    mlflow.set_experiment("JointAutoencoder-Testes")
    title = (
        f"MSE{mse_weight}_SSIM{ssim_weight}"
        f"_PSNR{psnr_weight}_NCC{ncc_weight}_VIF{vif_weight}_SCC{scc_weight}"
        f"_REC{rec_weight}"
    )
    title0 = f"JointAutoencoder0{title}"
    title1 = f"JointAutoencoder1{title}"
    
    with mlflow.start_run(run_name=title):
        
        mlflow.log_param("mse_weight", mse_weight)
        mlflow.log_param("ssim_weight", ssim_weight)
        mlflow.log_param("psnr_weight", psnr_weight)
        mlflow.log_param("ncc_weight", ncc_weight)
        mlflow.log_param("vif_weight", vif_weight)
        mlflow.log_param("scc_weight", scc_weight)
        mlflow.log_param("rec_weight", rec_weight)
        
        joint_0 = JointAutoencoder0().to(device)
        joint_1 = JointAutoencoder1().to(device)

        parameters = list(joint_0.parameters()) + list(joint_1.parameters())

        optimizer = torch.optim.Adam(parameters)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        epoch_pbar = tqdm(range(num_epochs), desc=f"Epochs [{title}]", leave=True)
        for epoch in epoch_pbar:

            joint_0.train()
            joint_1.train()

            running_loss = 0.0

            train_pbar = tqdm(train_loader, desc=f"Train epoch {epoch + 1}/{num_epochs}", leave=False)
            for x, y, _ in train_pbar:

                x = x.to(device)
                y = y.to(device)

                z0 = joint_0.encoder(x)
                if isinstance(z0, (tuple, list)):
                    recon0 = joint_0.decoder(*z0)
                else: 
                    recon0 = joint_0.decoder(z0)
                

                z1 = joint_1.encoder(x)
                if isinstance(z1, (tuple, list)):
                    recon1 = joint_1.decoder(*z1)
                else:
                    recon1 = joint_1.decoder(z1)

                loss = combined_loss(
                    recon0,
                    recon1,
                    z0,
                    z1,
                    y,
                    mse_weight,
                    ssim_weight,
                    psnr_weight,
                    ncc_weight,
                    vif_weight,
                    scc_weight,
                    rec_weight
                )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(parameters, 1.0)

                optimizer.step()

                running_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            running_loss /= len(train_loader)
            mlflow.log_metric("train_loss", running_loss, step=epoch)


            scheduler.step()

            # validation

            joint_0.eval()
            joint_1.eval()

            running_val_loss = 0.0

            with torch.no_grad():

                valid_pbar = tqdm(valid_loader, desc=f"Valid epoch {epoch + 1}/{num_epochs}", leave=False)
                for x, y, _ in valid_pbar:

                    x = x.to(device)
                    y = y.to(device)

                    recon0 = joint_0(x)
                    recon1 = joint_1(x)

                    z0 = joint_0.encoder(x)
                    z1 = joint_1.encoder(x)

                    val_loss = combined_loss(
                        recon0,
                        recon1,
                        z0,
                        z1,
                        y,
                        mse_weight,
                        ssim_weight,
                        psnr_weight,
                        ncc_weight,
                        vif_weight,
                        scc_weight,
                        rec_weight
                    )

                    running_val_loss += val_loss.item()
                    valid_pbar.set_postfix(val_loss=f"{val_loss.item():.4f}")

            val_epoch_loss = running_val_loss / len(valid_loader)
            epoch_pbar.set_postfix(train=f"{running_loss:.4f}", valid=f"{val_epoch_loss:.4f}")

        
        joint_0.eval()
        joint_1.eval()
        
        metrics_sum = {
            "MSE": 0.0,
            "SSIM": 0.0,
            "PSNR": 0.0,
            "NCC": 0.0,
            "VIF": 0.0,
            "SCC": 0.0
        }
        
        with torch.no_grad():

            test_metrics_pbar = tqdm(test_loader, desc="Test metrics", leave=False)
            for x, _, _ in test_metrics_pbar:

                x = x[:8].to(device)

                recon0 = joint_0(x)
                recon1 = joint_1(x)

                images_cpu = x.detach().cpu()
                recon0_cpu = recon0.detach().cpu()
                recon1_cpu = recon1.detach().cpu()

                images_dn = torch.clamp(denormalize(images_cpu), 0, 1)
                recon0_dn = torch.clamp(denormalize(recon0_cpu), 0, 1)
                recon1_dn = torch.clamp(denormalize(recon1_cpu), 0, 1)

                batch_metrics0 = calculate_all_metrics_torch(images_dn, recon0_dn)
                batch_metrics1 = calculate_all_metrics_torch(images_dn, recon1_dn)

                for k in metrics_sum:
                    metrics_sum[k] += 0.5 * (batch_metrics0[k] + batch_metrics1[k])
                test_metrics_pbar.set_postfix(mse=f"{batch_metrics0['MSE']:.4f}")


        metrics_avg = {k: float(v / len(test_loader)) for k, v in metrics_sum.items()}
        for name, value in metrics_avg.items():
            mlflow.log_metric(f"test_{name.lower()}", value)
        

        rec0 = plot_reconstruction(
            x,
            recon0,
            title0,
            "PUC",
            save_path="./results/Joint-Grid-Search"
        )

        rec1 = plot_reconstruction(
            x,
            recon1,
            title1,
            "PUC",
            save_path="./results/Joint-Grid-Search"
        )

        
        representations0 = []
        representations1 = []
        y_true = []

        with torch.no_grad():

            for img, label, _ in tqdm(test_loader, desc='CLUSTER'):

                img = img.to(device)

                z0 = joint_0.encoder(img)
                z1 = joint_1.encoder(img)

                representations0.append(z0.cpu())
                representations1.append(z1.cpu())

                y_true.extend(label.numpy())


        os.makedirs("./results/Joint-Grid-Search/Clusters/", exist_ok=True)

        path_cluster = (
            f'./results/Joint-Grid-Search/Clusters/'
            f'MSE{mse_weight}_SSIM{ssim_weight}_'
            f'PSNR{psnr_weight}_NCC{ncc_weight}_VIF{vif_weight}_SCC{scc_weight}_'
            f'REC{rec_weight}.png'
        )
        mean_distance = plot_cluster(
            representations0,
            representations1,
            y_true,
            path_cluster
        )

        del joint_0
        del joint_1
        torch.cuda.empty_cache()
        
        mlflow.log_artifact(rec0, artifact_path="reconstructions")
        mlflow.log_artifact(rec1, artifact_path="reconstructions")
        mlflow.log_artifact(path_cluster, artifact_path="cluster")
        mlflow.log_metric('val_loss', val_epoch_loss)
        mlflow.log_metric('mean_distance', mean_distance)


        return val_epoch_loss, mean_distance

# GRID SEARCH

mse_weights = [0.4, 0.5, 0.6, 0.7, 0.8]
ssim_weights = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
psnr_weights = [0.1, 0.2, 0.3]
ncc_weights = [0.1, 0.2, 0.3]
vif_weights = [0.1, 0.2, 0.3]
scc_weights = [0.1, 0.2, 0.3]
rec_weights = [0.3, 0.5, 0.7, 1.0]

results = []

total_combinations = (
    len(mse_weights)
    * len(ssim_weights)
    * len(psnr_weights)
    * len(ncc_weights)
    * len(vif_weights)
    * len(scc_weights)
    * len(rec_weights)
)

# Escreve resultados incrementalmente para não perder progresso em interrupções.
csv_path = "grid_search_results.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "MSE_Weight",
            "SSIM_Weight",
            "PSNR_Weight",
            "NCC_Weight",
            "VIF_Weight",
            "SCC_Weight",
            "rec_weights",
            "Val_Loss",
            "mean_distance",
        ]
    )

    best_result = None
    grid_pbar = tqdm(
        product(
            mse_weights,
            ssim_weights,
            psnr_weights,
            ncc_weights,
            vif_weights,
            scc_weights,
            rec_weights,
        ),
        total=total_combinations,
        desc="Grid Search",
        leave=True,
    )

    for mse_w, ssim_w, psnr_w, ncc_w, vif_w, scc_w, rec in grid_pbar:
        val_loss, mean_distance = train_models(
            mse_w,
            ssim_w,
            psnr_w,
            ncc_w,
            vif_w,
            scc_w,
            rec,
        )

        row = (mse_w, ssim_w, psnr_w, ncc_w, vif_w, scc_w, rec, val_loss, mean_distance)
        results.append(row)
        writer.writerow(row)
        csvfile.flush()

        if (best_result is None) or (val_loss < best_result[-2]):
            best_result = row

        grid_pbar.set_postfix(
            mse=mse_w,
            ssim=ssim_w,
            psnr=psnr_w,
            ncc=ncc_w,
            vif=vif_w,
            scc=scc_w,
            rec=rec,
            val=f"{val_loss:.4f}",
            best_val=f"{best_result[-2]:.4f}",
            dist=f"{mean_distance:.4f}",
        )
