import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

import mlflow

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from src.utils.datasets import CustomImageDataset
from src.models.joint_autoencoder import JointAutoencoder0, JointAutoencoder1
from src.config.config import Config
from src.utils.plot import plot_reconstruction
from src.models.loss.autoencoder_loss import ssim_loss, orthogonal_loss

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


# LOSS

def combined_loss(
    recon0,
    recon1,
    z0,
    z1,
    target,
    mse_weight=0.6,
    ssim_weight=0.4,
    rec_weight=0.6
):

    mse0 = criterion(recon0, target)
    mse1 = criterion(recon1, target)

    ssim0 = ssim_loss(recon0, target)
    ssim1 = ssim_loss(recon1, target)

    rec0 = mse_weight * mse0 + ssim_weight * ssim0
    rec1 = mse_weight * mse1 + ssim_weight * ssim1

    reconstruction_loss = (rec0 + rec1) / 2

    ort_loss = orthogonal_loss([z0, z1])

    total_loss = reconstruction_loss + rec_weight * ort_loss

    return total_loss



def train_models(mse_weight, ssim_weight, rec_weight, num_epochs=50):

    
    mlflow.set_experiment("JointAutoencoder-Testes-2")
    title =  f"MSE{mse_weight}_SSIM{ssim_weight}_REC{rec_weight}"
    title0 = f"JointAutoencoder0{title}"
    title1 = f"JointAutoencoder1{title}"
    
    with mlflow.start_run(run_name=title):
        
        mlflow.log_param("mse_weight", mse_weight)
        mlflow.log_param("ssim_weight", ssim_weight)
        mlflow.log_param("rec_weight", rec_weight)
        
        joint_0 = JointAutoencoder0().to(device)
        joint_1 = JointAutoencoder1().to(device)

        parameters = list(joint_0.parameters()) + list(joint_1.parameters())

        optimizer = torch.optim.Adam(parameters)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        for epoch in range(num_epochs):

            joint_0.train()
            joint_1.train()

            running_loss = 0.0

            for x, y, _ in train_loader:

                x = x.to(device)
                y = y.to(device)

                z0 = joint_0.encoder(x)
                if len(z0) == 3 :
                    recon0 = joint_0.decoder(*z0)
                else: 
                    recon0 = joint_0.decoder(z0)
                

                z1 = joint_1.encoder(x)
                if len(z1) == 3:
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
                    rec_weight
                )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(parameters, 1.0)

                optimizer.step()

                running_loss += loss.item()

            running_loss /= len(train_loader)
            mlflow.log_metric("train_loss", running_loss, step=epoch)


            scheduler.step()

            # validation

            joint_0.eval()
            joint_1.eval()

            running_val_loss = 0.0

            with torch.no_grad():

                for x, y, _ in valid_loader:

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
                        rec_weight
                    )

                    running_val_loss += val_loss.item()

            val_epoch_loss = running_val_loss / len(valid_loader)


            # RECONSTRUCTION PLOTS
        
        joint_0.eval()
        joint_1.eval()

        with torch.no_grad():

            for x, _, _ in test_loader:

                x = x[:8].to(device)

                recon0 = joint_0(x)
                recon1 = joint_1(x)

                break

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


            # CLUSTER REPRESENTATIONS
        
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

        path_cluster = f'./results/Joint-Grid-Search/Clusters/MSE{mse_weight}_SSIM{ssim_weight}_REC{rec_weight}.png'
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
mlflow.set_tracking_uri(Config().IP_LOCAL)
mse_weights = [0.4, 0.5, 0.6, 0.7, 0.8]
ssim_weights = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
rec_weights = [0.3, 0.5, 0.7, 1.0]

results = []

for mse_w in mse_weights:
    for ssim_w in ssim_weights:
        for rec in rec_weights:

            val_loss, mean_distance = train_models(
                mse_w,
                ssim_w,
                rec
            )

            results.append(
                (mse_w, ssim_w, rec, val_loss, mean_distance)
            )

            print(
                f"MSE: {mse_w}, SSIM: {ssim_w}, REC: {rec} "
                f"Val Loss: {val_loss:.4f}, Mean Distance: {mean_distance:.4f}"
            )


# SAVE RESULTS

with open("grid_search_results.csv", "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow(
        [
            "MSE_Weight",
            "SSIM_Weight",
            "rec_weights",
            "Val_Loss",
            "mean_distance"
        ]
    )

    for row in results:
        writer.writerow(row)
