import argparse
import torch
import torch.nn as nn
import os 
import pandas as pd
import datetime 
import mlflow 
from tqdm import tqdm 

from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset, DataLoader
from src.models.moe import SparseMoE
from src.utils.datasets import CustomImageDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.config import Config

config = Config()

model = SparseMoE(n_experts=4, top_k=2)

class ImageDataset(Dataset):
    def __init__(self, csv):
        self.data = pd.read_csv(csv)

        self.transform = transforms.Compose([
            transforms.Resize((124, 124)),   # garante 124x124
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], #padrão ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]

        image = Image.open(img_path).convert('RGB')
        image = np.array(image).transpose(2, 0, 1)  # [H,W,C] -> [C,H,W]
        image = torch.from_numpy(image).float() / 255.0

        image = self.transform(image)

        label = int(self.data.iloc[idx, 1])

        return image, label, idx


data_train = ImageDataset(csv='/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/batches/batch-64.csv')
data_val = ImageDataset(csv='/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/PUC_validation.csv')
data_test = ImageDataset(csv='/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/PUC_test.csv')

train_loader = DataLoader(data_train, num_workers=2, pin_memory=True)
val_loader = DataLoader(data_val, num_workers=2, pin_memory=True)
test_loader = DataLoader(data_test, num_workers=2, pin_memory=True)


mlflow.set_tracking_uri(config.IP_LOCAL)
mlflow.set_experiment('Moe')
    
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model.to(device)

with mlflow.start_run(run_name='MOE'):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(
        range(10),
        desc=f'[GPU] {0} Epochs',
        position=0
    ):
        
        model.train()
        model.float() 
        train_loss, train_correct, train_total = 0, 0, 0

        for x, y, idx in tqdm(train_loader, desc='Train'):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out, balance_loss = model(x)
                task_loss = criterion(out, y)
                loss = task_loss + 0.01 * balance_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_acc = 100 * train_correct / train_total
        train_avg_loss = train_loss / len(train_loader)


        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc='Val'):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    out, balance_loss = model(x)
                    task_loss = criterion(out, y)
                    loss = task_loss + 0.01 * balance_loss

                val_loss += loss.item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y, idx in tqdm(test_loader, desc='Test'):
                x = x.to(device)
                y = y.to(device)

                with torch.cuda.amp.autocast():
                    out, balance_loss = model(x)

                preds = out.argmax(1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                probs = torch.softmax(out, dim=1)

        mlflow.log_metric("test_acc", accuracy_score(y_true, y_pred))