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
from src.models.joint_autoencoder import JointAutoencoder0, JointAutoencoder1
from src.config.config import Config
from src.utils.plot import plot_reconstruction

import torch.nn.init as init

#ref: https://www.codegenes.net/blog/how-to-do-joint-training-on-many-models-pytorch/

train = CustomImageDataset(csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/batches/batch-1024.csv", autoencoder=True)
valid = CustomImageDataset(csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/PUC_validation.csv", autoencoder=True)
test = CustomImageDataset(csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/PUC_test.csv", autoencoder=True)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid, batch_size=32, shuffle=False)
test_loader = DataLoader(test, batch_size=32, shuffle=False)


config = Config()
device = config.DEVICE0
 
joint_0 = JointAutoencoder0()
joint_1 = JointAutoencoder1()

joint_0.to(device)
joint_1.to(device)


criterion_model_0 = nn.MSELoss()
criterion_model_1 = nn.MSELoss()

from src.models.loss.autoencoder_loss import ssim_loss, euclidean_distance_loss
 
def combined_loss(output_model_0, output_model_1, target):
    loss_mse_0 = criterion_model_0(output_model_0, target)
    loss_mse_1 = criterion_model_1(output_model_1, target)

    ssim_loss_value_0 = ssim_loss(output_model_0, target)
    ssim_loss_value_1 = ssim_loss(output_model_1, target)
    
    euclidean_distance_loss_value = euclidean_distance_loss(output_model_0, output_model_1)
    
    total_loss = ((loss_mse_0 * 0.6 + ssim_loss_value_0 * 0.4) + (loss_mse_1 * 0.6 + ssim_loss_value_1 * 0.4)) * 0.5 + euclidean_distance_loss_value * 0.5
    return total_loss

 
# Define optimizer
parameters = list(joint_0.parameters()) + list(joint_1.parameters())
optimizer = torch.optim.Adam(parameters)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
 
 
num_epochs = 20
for epoch in tqdm(range(num_epochs), desc='Epochs'):
    
    joint_0.train()
    joint_1.train()
    
    running_loss = 0.0
    
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        
        output_joint_0 = joint_0(x)
        output_joint_1 = joint_1(x)
    
        loss = combined_loss(output_joint_0, output_joint_1, y)
        running_loss += loss.item()

        optimizer.zero_grad()
 
        loss.backward()
    
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
    
        optimizer.step()
    
    epoch_loss = running_loss / len(train_loader)
    scheduler.step()
    
    joint_0.eval()
    joint_1.eval()
    
    running_val_loss = 0.0
    
    with torch.no_grad():
        for x, y, _ in valid_loader:
            x, y = x.to(device), y.to(device)
            
            output_joint_0 = joint_0(x)
            output_joint_1 = joint_1(x)
        
            val_loss = combined_loss(output_joint_0, output_joint_1, y)
            running_val_loss += val_loss.item()
    
    val_epoch_loss = running_val_loss / len(valid_loader)
 
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')


joint_0.eval()
joint_1.eval()

with torch.no_grad():
    for x, _, _ in test_loader:
        x = x[:8].to(device)
        recon0 = joint_0(x)
        recon1 = joint_1(x)
        break

plot_path = plot_reconstruction(
    x, recon0, "JointAutoencoder0", "PUC"
)

plot_path = plot_reconstruction(
    x, recon1, "JointAutoencoder1", "PUC"
)

