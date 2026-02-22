import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    image: Tensor [B, C, H, W] em CPU
    """

    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor")

    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std  = torch.tensor(std).view(1, -1, 1, 1)

    image = image * std + mean
    return torch.clamp(image, 0, 1)


def plot_reconstruction(original, reconstructed, model_name, dataset_name, save_path="./results/"):

    os.makedirs(save_path, exist_ok=True)

    # Garantir CPU e remover grad
    original = original.detach().cpu()
    if isinstance(reconstructed, tuple):
        reconstructed = reconstructed[0]
    reconstructed = reconstructed.detach().cpu()

    # Denormalizar
    original = denormalize(original)
    reconstructed = denormalize(reconstructed)

    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    fig.suptitle(f"{model_name} - {dataset_name} - Reconstruction", fontsize=16)

    for i in range(min(8, original.size(0))):

        axes[0, i].imshow(original[i].permute(1, 2, 0).numpy())
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).numpy())
        axes[1, i].axis("off")

    plt.tight_layout()

    filename = f"{save_path}/{model_name}_{dataset_name}_reconstruction.png"
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()

    return filename


import mlflow

def log_confusion_matrix_mlflow(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
    artifact_name="confusion_matrix.png",
    normalize=False
):
    """
    Confusion Matrix em tons de azul (estilo seaborn heatmap),
    usando apenas matplotlib e salvando no MLflow.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))

    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    # colorbar (igual seaborn)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count" if not normalize else "Proportion", rotation=-90, va="bottom")

    # valores nas células (com contraste automático)
    thresh = cm.max() / 2.0
    fmt = ".2f" if normalize else "d"

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11
            )

    ax.set_ylim(len(cm) - 0.5, -0.5)  # corrige corte do matplotlib
    plt.tight_layout()

    mlflow.log_figure(fig, artifact_name)
    plt.close(fig)

    return cm
