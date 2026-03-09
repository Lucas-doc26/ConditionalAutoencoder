import pandas as pd 
from src.utils.datasets import CustomImageDataset

from tqdm import tqdm
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import accuracy_score

from src.models.classifier import Classifier
from src.models.autoencoders_with_skip_connections import SkipEncoder0, SkipEncoder1
from src.config.config import Config

import torch.nn.init as init

#ref: https://www.codegenes.net/blog/how-to-do-joint-training-on-many-models-pytorch/

train = CustomImageDataset(csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/batches/batch-1024.csv", autoencoder=False)
valid = CustomImageDataset(csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/PUC_validation.csv", autoencoder=False)
test = CustomImageDataset(csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/PUC_test.csv", autoencoder=False)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid, batch_size=32, shuffle=False)
test_loader = DataLoader(test, batch_size=32, shuffle=False)



config = Config()
device = config.DEVICE0
 
 
skip_0 = SkipEncoder0()
skip_0.load_state_dict(torch.load('/home/lucas.ocunha/ConditionalAutoencoder/models/SkipAutoencoder0/CNR/encoder.pth', map_location=device))
skip_0.to(device)
model_0 = Classifier(skip_0, latent_dim=skip_0.latent_dim, num_classes=2)
model_0.to(device)

skip_1 = SkipEncoder1()
skip_1.load_state_dict(torch.load('/home/lucas.ocunha/ConditionalAutoencoder/models/SkipAutoencoder1/CNR/encoder.pth', map_location=device))
skip_1.to(device)
model_1 = Classifier(skip_1, latent_dim=skip_1.latent_dim, num_classes=2)
model_1.to(device) 

 
# Initialize parameters
 
for m in model_0.modules():
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
 
# Define loss function
criterion_cnn = nn.CrossEntropyLoss()
criterion_rnn = nn.CrossEntropyLoss()
 
 
def combined_loss(output_model_0, output_model_1, target):
    loss_cnn = criterion_cnn(output_model_0, target)
    loss_rnn = criterion_rnn(output_model_1, target)
    return loss_cnn + loss_rnn
 
 
# Define optimizer
parameters = list(model_0.parameters()) + list(model_1.parameters())
optimizer = optim.Adam(parameters, lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
 
 
num_epochs = 100
best_val_loss = float('inf')
patience = 5
counter = 0
for epoch in range(num_epochs):
    
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        
        output_model_0 = model_0(x)
        output_model_1 = model_1(x)
    
        loss = combined_loss(output_model_0, output_model_1, y)

        optimizer.zero_grad()
 
        
        loss.backward()
    
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
    
        optimizer.step()
        scheduler.step()
    
    
    for x, y, _ in valid_loader:
        x, y = x.to(device), y.to(device)
        
        output_model_0 = model_0(x)
        output_model_1 = model_1(x)
    
        val_loss = combined_loss(output_model_0, output_model_1, y)
 
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')


y_pred_model_0 = []
y_pred_model_1 = []
y_true = pd.read_csv("/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/PUC_test.csv")["class"].values.tolist()
model_0.eval()
model_1.eval()
with torch.no_grad():
    for x, y, _ in tqdm(test_loader, desc='Test'):
        x, y = x.to(device), y.to(device)
        
        output_model_0 = model_0(x)
        output_model_1 = model_1(x)
    
        y_pred_model_0.extend(output_model_0.argmax(dim=1).cpu().numpy())
        y_pred_model_1.extend(output_model_1.argmax(dim=1).cpu().numpy())


print("Model 0 Accuracy:", accuracy_score(y_true, y_pred_model_0))
print("Model 1 Accuracy:", accuracy_score(y_true, y_pred_model_1))  