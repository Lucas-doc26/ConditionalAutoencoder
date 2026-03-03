import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared Backbone (feature extractor)
# ============================================================

class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 124 -> 62

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> [B, 64, 1, 1]
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # [B, 64]


# ============================================================
# Expert
# ============================================================

class Expert(nn.Module):
    def __init__(self, in_features=64, num_classes=2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# ============================================================
# Router (Top-k + Load Balancing)
# ============================================================

class Router(nn.Module):
    def __init__(self, in_features, n_experts, top_k=2, temperature=1.5):
        super().__init__()

        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature

        self.linear = nn.Linear(in_features, n_experts)

    def forward(self, x):
        """
        x: [B, in_features]
        """

        logits = self.linear(x) / self.temperature
        probs = F.softmax(logits, dim=1)  # distribuição completa

        top_k_probs, top_k_indices = torch.topk(
            probs, self.top_k, dim=1
        )

        gates = torch.zeros_like(probs)
        gates.scatter_(1, top_k_indices, top_k_probs)

        # --------------------------
        # Load balancing loss
        # --------------------------

        importance = probs.mean(0)              # prob média por expert
        load = (gates > 0).float().mean(0)      # frequência de uso

        balance_loss = (importance * load).sum() * self.n_experts

        return gates, top_k_indices, balance_loss


# ============================================================
# Sparse Mixture of Experts
# ============================================================

class SparseMoE(nn.Module):
    def __init__(self,
                 n_experts=4,
                 top_k=2,
                 num_classes=10,
                 temperature=1.5):

        super().__init__()

        self.n_experts = n_experts
        self.top_k = top_k

        self.backbone = SharedBackbone()

        self.router = Router(
            in_features=64,
            n_experts=n_experts,
            top_k=top_k,
            temperature=temperature
        )

        self.experts = nn.ModuleList([
            Expert(in_features=64, num_classes=num_classes)
            for _ in range(n_experts)
        ])

    def forward(self, x):

        B = x.size(0)
        device = x.device

        features = self.backbone(x)  # [B, 64]

        gates, top_k_indices, balance_loss = self.router(features)

        final_output = torch.zeros(
            B,
            self.experts[0].classifier[-1].out_features,
            device=device
        )

        # ====================================================
        # Sparse execution real
        # ====================================================

        for expert_id in range(self.n_experts):

            mask = (top_k_indices == expert_id).any(dim=1)

            if mask.sum() == 0:
                continue

            selected_features = features[mask]
            selected_gates = gates[mask, expert_id]

            expert_out = self.experts[expert_id](selected_features)

            expert_out = expert_out * selected_gates.unsqueeze(1)

            final_output[mask] += expert_out

        return final_output, balance_loss