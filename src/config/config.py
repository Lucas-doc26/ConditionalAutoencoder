import torch
import numpy as np


def latent_dims(n_model, range_latent=(256, 512)):
    rng = np.random.default_rng(n_model)
    return rng.integers(range_latent[0], range_latent[1])


class Config():
    def __init__(self, range_latent=(256, 512)):
        self.IP_LOCAL = "http://127.0.0.1:5000"
        self.DEVICE0 = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.DEVICE1 = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.DEVICES = [self.DEVICE0, self.DEVICE1]
        self.LATENT_DIMS = [latent_dims(i, range_latent) for i in range(10)]
