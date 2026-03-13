from src.utils.image_metrics import calculate_ssim_torch
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances


def ssim_loss(recon_x, x):
    ssim = calculate_ssim_torch(recon_x, x)
    return 1 - ssim


# Função que calcula a distância euclidiana entre os embeddings de dois clusters
def euclidean_distance_loss(cluster_a, cluster_b):

    a = cluster_a.view(cluster_a.size(0), -1)
    b = cluster_b.view(cluster_b.size(0), -1)

    distances = torch.norm(a - b, dim=1)

    mean_distance = distances.mean()

    normalized = torch.tanh(mean_distance)

    return -normalized


def orthogonal_loss(z0, z1):

    z0 = z0.view(z0.size(0), -1)
    z1 = z1.view(z1.size(0), -1)

    z0 = F.normalize(z0, dim=1)
    z1 = F.normalize(z1, dim=1)

    cosine = (z0 * z1).sum(dim=1)

    loss = (cosine ** 2).mean()

    return loss
