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

from src.models.loss.autoencoder_loss import ssim_loss, euclidean_distance_loss, orthogonal_loss
 
def combined_loss(output_model_0,
                  output_model_1,
                  target,
                  mse_weight=0.6,
                  ssim_weight=0.4,
                  rec_weight=0.6):

    # reconstruction losses
    mse0 = criterion_model_0(output_model_0, target)
    mse1 = criterion_model_1(output_model_1, target)

    ssim0 = ssim_loss(output_model_0, target)
    ssim1 = ssim_loss(output_model_1, target)

    rec0 = mse_weight * mse0 + ssim_weight * ssim0
    rec1 = mse_weight * mse1 + ssim_weight * ssim1

    reconstruction_loss = (rec0 + rec1) / 2


    orthogonal_loss = orthogonal_loss(output_model_0, output_model_1)

    total_loss =  reconstruction_loss + rec_weight * orthogonal_loss

    return total_loss

def train_models(mse_weight, ssim_weight, rec_weight, num_epochs=100):
    joint_0 = JointAutoencoder0()
    joint_1 = JointAutoencoder1()

    joint_0.to(device)
    joint_1.to(device)

    parameters = list(joint_0.parameters()) + list(joint_1.parameters())
    optimizer = torch.optim.Adam(parameters)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(num_epochs):
        joint_0.train()
        joint_1.train()
        
        running_loss = 0.0
        
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            
            output_joint_0 = joint_0(x)
            output_joint_1 = joint_1(x)
        
            loss = combined_loss(output_joint_0, output_joint_1, y, mse_weight, ssim_weight, rec_weight)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
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
            
                val_loss = combined_loss(output_joint_0, output_joint_1, y, mse_weight, ssim_weight, rec_weight)
                running_val_loss += val_loss.item()
        
        val_epoch_loss = running_val_loss / len(valid_loader)
    
    # Plot reconstructions
    joint_0.eval()
    joint_1.eval()
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x[:8].to(device)
            recon0 = joint_0(x)
            recon1 = joint_1(x)
            break

    title0 = f"JointAutoencoder0_MSE{mse_weight}_SSIM{ssim_weight}_REC{rec_weight}"
    title1 = f"JointAutoencoder1_MSE{mse_weight}_SSIM{ssim_weight}_REC{rec_weight}"
    
    plot_reconstruction(x, recon0, title0, "PUC", save_path='./results/Joint-Grid-Search')
    plot_reconstruction(x, recon1, title1, "PUC", save_path='./results/Joint-Grid-Search')
    
    return val_epoch_loss

# Grid search with more values (25 combinations)
mse_weights = [0.4, 0.5, 0.6, 0.7, 0.8]
ssim_weights = [0.2, 0.3, 0.4, 0.5, 0.6]
rec_weights = [0.3, 0.5, 0.7, 1.0]

results = []

for mse_w in mse_weights:
    for ssim_w in ssim_weights:
        for rec in rec_weights
            val_loss = train_models(mse_w, ssim_w, rec)
            results.append((mse_w, ssim_w, rec, val_loss))
            print(f'MSE: {mse_w}, SSIM: {ssim_w}, Val Loss: {val_loss:.4f}')

# Save results to CSV
import csv
with open('grid_search_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['MSE_Weight', 'SSIM_Weight', "rec_weights", 'Val_Loss'])
    for row in results:
        writer.writerow(row)

