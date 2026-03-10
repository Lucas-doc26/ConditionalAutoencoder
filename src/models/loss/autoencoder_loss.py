from src.utils.image_metrics import calculate_ssim_torch
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances


def ssim_loss(recon_x, x):
    ssim = calculate_ssim_torch(recon_x, x)
    return 1 - ssim


# Função que calcula a distância euclidiana entre os embeddings de dois clusters
# 0 -> são iguais
# 1 -> são completamente diferentes
# como quero variar, faço -1 para inverter a escala
def euclidean_distance_loss(cluster_a, cluster_b):
    # Calcula a distância por amostra no batch
    batch_size = cluster_a.shape[0]
    distances = []
    for i in range(batch_size):
        dist = torch.dist(cluster_a[i].flatten(), cluster_b[i].flatten())
        distances.append(dist)
    mean_distance = torch.stack(distances).mean()
    
    normalized = mean_distance / 2
    return -1 - normalized