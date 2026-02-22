import torch
import numpy as np


def latent_dims(n_model):
    rng = np.random.default_rng(n_model)
    return rng.integers(256, 512)


class Config:
    IP_LOCAL = "http://127.0.0.1:5000"
    DEVICE0 = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE1 = "cuda:1" if torch.cuda.is_available() else "cpu"
    DEVICES = [DEVICE0, DEVICE1]
    LATENT_DIMS = [latent_dims(i) for i in range(10)]
