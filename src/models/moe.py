import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
from torchvision import transforms
from PIL import Image

class ExpertCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 124 -> 62
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(64 * 62 * 62, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Router(nn.Module):
    def __init__(self, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(3, n_experts)

    def forward(self, x):
        # x: [B, C, H, W]
        x_pooled = self.pool(x).view(x.size(0), -1)  # [B, 3]
        logits = self.linear(x_pooled)  # [B, n_experts]
        
        # seleciona os top_k melhores experts por amostra
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)  # [B, top_k]
        
        # Normaliza apenas os top-k
        top_k_gates = F.softmax(top_k_logits, dim=1)  # [B, top_k]
        
        # Cria matriz de gates completa (zeros para não-selecionados)
        gates = torch.zeros_like(logits)  # [B, n_experts]
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices


class MoECNN(nn.Module):
    def __init__(self, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([ExpertCNN() for _ in range(n_experts)])
        self.router = Router(n_experts, top_k=top_k)

    def forward(self, x):
        # Router seleciona top-k experts
        gates, top_k_indices = self.router(x)  # gates: [B, n_experts], top_k_indices: [B, top_k]
        
        # Executa apenas os experts selecionados (mais eficiente)
        batch_size = x.size(0)
        expert_outputs = []
        
        for i in range(self.n_experts):
            expert_outputs.append(self.experts[i](x))
        
        outputs = torch.stack(expert_outputs, dim=1)  # [B, n_experts, C]
        
        # Aplica gates (ponderacao dos experts)
        gates = gates.unsqueeze(-1)  # [B, n_experts, 1]
        weighted_outputs = outputs * gates  # [B, n_experts, C]
        
        return weighted_outputs.sum(dim=1)  # [B, C]

