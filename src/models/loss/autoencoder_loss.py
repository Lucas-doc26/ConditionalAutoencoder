from src.utils.image_metrics import calculate_ssim_torch
import torch
import torch.nn.functional as F


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


"""def orthogonal_loss(z0, z1):

    z0 = z0.view(z0.size(0), -1)
    z1 = z1.view(z1.size(0), -1)

    z0 = F.normalize(z0, dim=1)
    z1 = F.normalize(z1, dim=1)

    cosine = (z0 * z1).sum(dim=1)

    loss = (cosine ** 2).mean()

    return loss"""

#v2
def orthogonal_loss(embeddings: list):
    """
    Abordagem matricial: O(K²D) em vez de O(K² * chamadas)
    """
    # Stack: (K, B, D) → média no batch → (K, D)
    zs = torch.stack([
        F.normalize(z.view(z.size(0), -1), dim=1).mean(dim=0) 
        for z in embeddings
    ])  # (K, D)
    
    gram = zs @ zs.T          # (K, K) — produto interno entre todas as labels
    I = torch.eye(len(embeddings), device=gram.device)
    loss = (gram - I).pow(2).sum()  # penaliza off-diagonal
    return loss