import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd 
import numpy as np 
import cv2 

import torch.nn as nn
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder0(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc_fc = nn.Linear(16 * 32 * 32, 1849)

        # Decoder
        self.dec_fc = nn.Linear(1849, 16 * 32 * 32)
        self.dec_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(8, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_conv1(x))   # (B, 8, 128, 128)
        x = self.pool(x)                # (B, 8, 64, 64)
        x = F.relu(self.enc_conv2(x))   # (B, 16, 64, 64)
        x = self.pool(x)                # (B, 16, 32, 32)
        x = torch.flatten(x, 1)         # (B, 16384)
        z = self.enc_fc(x)              # (B, 1849)

        # Decoder
        x = self.dec_fc(z)              # (B, 16384)
        x = x.view(-1, 16, 32, 32)
        x = F.relu(self.dec_conv1(x))   # (B, 16, 32, 32)
        x = self.upsample(x)            # (B, 16, 64, 64)
        x = F.relu(self.dec_conv2(x))   # (B, 8, 64, 64)
        x = self.upsample(x)            # (B, 8, 128, 128)
        x = torch.sigmoid(self.dec_conv3(x))  # (B, 3, 128, 128)

        return x

class Autoencoder1(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 8, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(8, 64, 3, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc_fc = nn.Linear(32 * 16 * 16, 467)

        # Decoder
        self.dec_fc = nn.Linear(467, 32 * 16 * 16)

        self.dec_conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(64, 8, 3, padding=1)
        self.dec_conv4 = nn.Conv2d(8, 32, 3, padding=1)
        self.dec_conv5 = nn.Conv2d(32, 3, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    # Encoder
    def encode(self, x):
        x = F.relu(self.enc_conv1(x))   # (B, 32, 128, 128)
        x = self.pool(x)                # (B, 32, 64, 64)

        x = F.relu(self.enc_conv2(x))   # (B, 8, 64, 64)
        x = self.pool(x)                # (B, 8, 32, 32)

        x = F.relu(self.enc_conv3(x))   # (B, 64, 32, 32)
        x = self.pool(x)                # (B, 64, 16, 16)

        x = F.relu(self.enc_conv4(x))   # (B, 32, 16, 16)

        x = torch.flatten(x, 1)         # (B, 8192)
        z = self.enc_fc(x)              # (B, 467)
        return z

    # Decoder
    def decode(self, z):
        x = self.dec_fc(z)              # (B, 8192)
        x = x.view(-1, 32, 16, 16)

        x = F.relu(self.dec_conv1(x))   # (B, 32, 16, 16)
        x = F.relu(self.dec_conv2(x))   # (B, 64, 16, 16)

        x = self.upsample(x)            # (B, 64, 32, 32)
        x = F.relu(self.dec_conv3(x))   # (B, 8, 32, 32)

        x = self.upsample(x)            # (B, 8, 64, 64)
        x = F.relu(self.dec_conv4(x))   # (B, 32, 64, 64)

        x = self.upsample(x)            # (B, 32, 128, 128)
        x = torch.sigmoid(self.dec_conv5(x))  # (B, 3, 128, 128)

        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)



class Autoencoder2(nn.Module):
    def __init__(self):
        super().__init__()

        # ======================
        # Encoder
        # ======================
        self.enc_conv1 = nn.Conv2d(3, 32, 3, padding=1)    # 896
        self.enc_conv2 = nn.Conv2d(32, 32, 3, padding=1)   # 9248
        self.enc_conv3 = nn.Conv2d(32, 16, 3, padding=1)   # 4624

        self.enc_conv4 = nn.Conv2d(16, 16, 3, padding=1)   # 2320
        self.enc_conv5 = nn.Conv2d(16, 16, 3, padding=1)   # 2320

        self.pool = nn.MaxPool2d(2, 2)

        # 16 * 16 * 16 = 4096
        self.enc_fc = nn.Linear(4096, 1411)

        # ======================
        # Decoder
        # ======================
        self.dec_fc = nn.Linear(1411, 4096)

        self.dec_conv1 = nn.Conv2d(16, 16, 3, padding=1)   # 2320
        self.dec_conv2 = nn.Conv2d(16, 16, 3, padding=1)   # 2320
        self.dec_conv3 = nn.Conv2d(16, 16, 3, padding=1)   # 2320

        self.dec_conv4 = nn.Conv2d(16, 32, 3, padding=1)   # 4640
        self.dec_conv5 = nn.Conv2d(32, 32, 3, padding=1)   # 9248
        self.dec_conv6 = nn.Conv2d(32, 3, 3, padding=1)    # 867

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    # ----------------------
    # Encoder
    # ----------------------
    def encode(self, x):
        x = F.relu(self.enc_conv1(x))   # (B, 32, 128, 128)
        x = F.relu(self.enc_conv2(x))   # (B, 32, 128, 128)
        x = F.relu(self.enc_conv3(x))   # (B, 16, 128, 128)

        x = self.pool(x)                # (B, 16, 64, 64)

        x = F.relu(self.enc_conv4(x))   # (B, 16, 64, 64)
        x = self.pool(x)                # (B, 16, 32, 32)

        x = F.relu(self.enc_conv5(x))   # (B, 16, 32, 32)
        x = self.pool(x)                # (B, 16, 16, 16)

        x = torch.flatten(x, 1)         # (B, 4096)
        z = self.enc_fc(x)              # (B, 1411)
        return z

    # ----------------------
    # Decoder
    # ----------------------
    def decode(self, z):
        x = self.dec_fc(z)              # (B, 4096)
        x = x.view(-1, 16, 16, 16)

        x = F.relu(self.dec_conv1(x))   # (B, 16, 16, 16)
        x = self.upsample(x)            # (B, 16, 32, 32)

        x = F.relu(self.dec_conv2(x))   # (B, 16, 32, 32)
        x = self.upsample(x)            # (B, 16, 64, 64)

        x = F.relu(self.dec_conv3(x))   # (B, 16, 64, 64)
        x = self.upsample(x)            # (B, 16, 128, 128)

        x = F.relu(self.dec_conv4(x))   # (B, 32, 128, 128)
        x = F.relu(self.dec_conv5(x))   # (B, 32, 128, 128)

        x = torch.sigmoid(self.dec_conv6(x))  # (B, 3, 128, 128)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class Autoencoder3(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc_fc = nn.Linear(8 * 16 * 16, 1674)

        # Decoder
        self.dec_fc = nn.Linear(1674, 8 * 16 * 16)
        self.dec_conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(8, 32, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(32, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        z = self.enc_fc(x)
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 8, 16, 16)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class Autoencoder4(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 16, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(16, 64, 3, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.enc_conv6 = nn.Conv2d(64, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc_fc = nn.Linear(16 * 8 * 8, 562)

        # Decoder
        self.dec_fc = nn.Linear(562, 16 * 8 * 8)
        self.dec_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.dec_conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_conv5 = nn.Conv2d(64, 16, 3, padding=1)
        self.dec_conv6 = nn.Conv2d(16, 64, 3, padding=1)
        self.dec_conv7 = nn.Conv2d(64, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv4(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv5(x))
        x = F.relu(self.enc_conv6(x))
        x = torch.flatten(x, 1)
        z = self.enc_fc(x)
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 16, 8, 8)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv3(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv4(x))
        x = F.relu(self.dec_conv5(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec_conv7(self.dec_conv6(x)))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class Autoencoder5(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 128, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc_fc = nn.Linear(8 * 8 * 8, 685)

        # Decoder
        self.dec_fc = nn.Linear(685, 8 * 8 * 8)
        self.dec_conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(8, 128, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(128, 16, 3, padding=1)
        self.dec_conv4 = nn.Conv2d(16, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        z = self.enc_fc(x)
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 8, 8, 8)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv3(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class Autoencoder6(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc_fc = nn.Linear(64 * 16 * 16, 1262)

        # Decoder
        self.dec_fc = nn.Linear(1262, 64 * 16 * 16)
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(64, 16, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(16, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        z = self.enc_fc(x)
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 64, 16, 16)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class Autoencoder7(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(128, 32, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc_fc = nn.Linear(16 * 32 * 32, 1960)

        # Decoder
        self.dec_fc = nn.Linear(1960, 16 * 32 * 32)
        self.dec_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(32, 128, 3, padding=1)
        self.dec_conv4 = nn.Conv2d(128, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        z = self.enc_fc(x)
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 16, 32, 32)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class Autoencoder8(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc_fc = nn.Linear(128 * 16 * 16, 838)

        # Decoder
        self.dec_fc = nn.Linear(838, 128 * 16 * 16)
        self.dec_conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(64, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        z = self.enc_fc(x)
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 128, 16, 16)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class Autoencoder9(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(8, 32, 3, padding=1)
        self.enc_conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.enc_conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.enc_fc = nn.Linear(32 * 4 * 4, 148)

        # Decoder
        self.dec_fc = nn.Linear(148, 32 * 4 * 4)
        self.dec_conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.dec_conv4 = nn.Conv2d(32, 8, 3, padding=1)
        self.dec_conv5 = nn.Conv2d(8, 16, 3, padding=1)
        self.dec_conv6 = nn.Conv2d(16, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv4(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv5(x))
        x = torch.flatten(x, 1)
        z = self.enc_fc(x)
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 32, 4, 4)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv3(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv4(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv5(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec_conv6(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

