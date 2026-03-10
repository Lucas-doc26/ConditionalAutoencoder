from src.utils.image_metrics import calculate_ssim_torch
from sklearn.metrics import silhouette_score
import numpy as np


def ssim_loss(recon_x, x):
    ssim = calculate_ssim_torch(recon_x, x)
    return 1 - ssim

def silhouette_loss(cluster_a, cluster_b):
    # cluster_a e cluster_b são tensores de forma (N, D)
    # onde N é o número de amostras e D é a dimensão das features
    # calculamos a silhouette score usando sklearn
    cluster_a_np = cluster_a.cpu().detach().numpy()
    cluster_b_np = cluster_b.cpu().detach().numpy()
    
    # Criamos um array de rótulos para as amostras
    labels = np.array([0] * len(cluster_a_np) + [1] * len(cluster_b_np))
    
    # Concatenamos os clusters para calcular a silhouette score
    combined_clusters = np.vstack((cluster_a_np, cluster_b_np))
    
    score = silhouette_score(combined_clusters, labels)
    
    return 1 - score  # Queremos minimizar a perda, então subtraímos de 1