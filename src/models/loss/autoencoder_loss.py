from src.utils.image_metrics import calculate_ssim_torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances


print("Distância média entre embeddings:", mean_distance)
def ssim_loss(recon_x, x):
    ssim = calculate_ssim_torch(recon_x, x)
    return 1 - ssim


# Função que calcula a distância euclidiana entre os embeddings de dois clusters
# 0 -> são iguais
# 1 -> são completamente diferentes
# como quero variar, faço -1 para inverter a escala
def euclidean_distance_loss(cluster_a, cluster_b):

    cluster_a = F.normalize(cluster_a, dim=1)
    cluster_b = F.normalize(cluster_b, dim=1)

    a = cluster_a.detach().cpu().numpy()
    b = cluster_b.detach().cpu().numpy()

    dist_matrix = euclidean_distances(a, b)
    mean_distance = dist_matrix.mean()

    normalized = mean_distance / 2

    return -1 - normalized