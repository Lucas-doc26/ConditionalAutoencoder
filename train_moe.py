import argparse
import torch
import torch.nn as nn
import os 
import pandas as pd
import datetime 
import mlflow 
import tqdm

from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset, DataLoader
from src.models.moe import MoECNN
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


model = MoECNN(n_experts=10, top_k=3)

data_train = CustomImageDataset(csv='/home/lucas/Representation-Fusion/CSV/PUC/PUC_.csv', autoencoder=False)
data_val = CustomImageDataset(csv='/home/lucas/Representation-Fusion/CSV/PUC/PUC_.csv', autoencoder=False)
data_test = CustomImageDataset(csv='/home/lucas/Representation-Fusion/CSV/PUC/PUC_.csv', autoencoder=False)

train_loader = DataLoader(data_train, num_worker=2, pin_memory=True)
val_loader = DataLoader(data_val, num_worker=2, pin_memory=True)
test_loader = DataLoader(data_test, num_worker=2, pin_memory=True)


mlflow.set_tracking_uri('http://136.248.106.79:5000')

    
device = "cuda:0" if torch.cuda.is_available() else "cpu"

with mlflow.start_run(run_name='Teste-MOE'):
    mlflow.log_input(data_train, context='Train')
    mlflow.log_input(data_val, context='Val')
    mlflow.log_input(data_test, context='Test')


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(
        range(10),
        desc=f'[GPU] {0} Epochs',
        position=0
    ):
        
        model.train()

        train_loss, train_correct, train_total = 0, 0, 0

        for x, y, idx in train_loader:
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
            for x, y, _ in val_loader:
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
            for x, y, idx in test_loader:
                x = x.to(device)
                y = y.to(device)

                with torch.cuda.amp.autocast():
                    out, balance_loss = model(x)

                preds = out.argmax(1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                probs = torch.softmax(out, dim=1)

        mlflow.log_metric("test_acc", accuracy_score(y_true, y_pred))