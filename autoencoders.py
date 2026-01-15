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

class Encoder0(nn.Module):
    def __init__(self, latent_dim=1849):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 8, 128, 128)
        x = self.pool(x)            # (B, 8, 64, 64)

        x = F.relu(self.conv2(x))   # (B, 16, 64, 64)
        x = self.pool(x)            # (B, 16, 32, 32)

        x = self.flatten(x)         # (B, 16384)
        x = self.fc(x)              # (B, 1849)
        return x
    
class Decoder0(nn.Module):
    def __init__(self, latent_dim=1849):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 32 * 32 * 16)

        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       # (B, 16384)
        x = x.view(-1, 16, 32, 32)           # (B, 16, 32, 32)

        x = F.relu(self.conv1(x))            # (B, 16, 32, 32)
        x = self.upsample(x)                 # (B, 16, 64, 64)

        x = F.relu(self.conv2(x))            # (B, 8, 64, 64)
        x = self.upsample(x)                 # (B, 8, 128, 128)

        x = self.conv3(x)                    # (B, 3, 128, 128)
        return x
    
class Autoencoder0(nn.Module):
    def __init__(self, latent_dim=1849):
        super().__init__()
        self.encoder = Encoder0(latent_dim)
        self.decoder = Decoder0(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

#################################
class Encoder1(nn.Module):
    def __init__(self, latent_dim=467):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 32, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 8, 128, 128)
        x = self.pool(x)            # (B, 8, 64, 64)

        x = F.relu(self.conv2(x))   # (B, 16, 64, 64)
        x = self.pool(x)            # (B, 16, 32, 32)

        x = F.relu(self.conv3(x))   # (B, 16, 64, 64)
        x = self.pool(x)           # (B, 16, 32, 32)
        
        x = F.relu(self.conv4(x))   # (B, 16, 64, 64)
        
        x = self.flatten(x)         # (B, 16384)
        x = self.fc(x)              # (B, 467)
        return x
    
class Decoder1(nn.Module):
    def __init__(self, latent_dim=467):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 16 * 16 * 32)

        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        x = self.fc(x)                       # (B, 8192)
        x = x.view(-1, 32, 16, 16)           # (B, 32, 16, 16)

        x = F.relu(self.conv1(x))            # (B, 32, 16, 16)
        x = F.relu(self.conv2(x))            # (B, 64, 16, 16)
        x = self.upsample(x)                 # (B, 64, 32, 32)

        x = F.relu(self.conv3(x))            # (B, 8, 32, 32)
        x = self.upsample(x)                 # (B, 8, 64, 64)

        x = F.relu(self.conv4(x))            # (B, 32, 64, 64)
        x = self.upsample(x)                 # (B, 32, 128, 128)

        x = self.conv5(x)                    # (B, 3, 128, 128)

        return x
    
class Autoencoder1(nn.Module):
    def __init__(self, latent_dim=467):
        super().__init__()
        self.encoder = Encoder1(latent_dim)
        self.decoder = Decoder1(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


#################################
class Encoder2(nn.Module):
    def __init__(self, latent_dim=1411):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 8, 128, 128)

        x = F.relu(self.conv2(x))   # (B, 16, 64, 64)

        x = F.relu(self.conv3(x))   # (B, 16, 64, 64)
        x = self.pool(x)            # (B, 16, 32, 32)
                
        x = F.relu(self.conv4(x))   # (B, 16, 64, 64)
        x = self.pool(x)            # (B, 16, 32, 32)
        
        x = F.relu(self.conv5(x))   # (B, 16, 64, 64)
        x = self.pool(x)            # (B, 16, 32,
        
        x = self.flatten(x)         # (B, 16384)
        x = self.fc(x)              # (B, 1411)
        return x
    
class Decoder2(nn.Module):
    def __init__(self, latent_dim=1411):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 16 * 16 * 16)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       # (B, 4096)
        x = x.view(-1, 16, 16, 16)           # (B, 16, 16, 16)

        x = F.relu(self.conv6(x))            # (B, 16, 16, 16)
        x = self.upsample(x)                 # (B, 16, 32, 32)

        x = F.relu(self.conv5(x))            # (B, 16, 32, 32)
        x = self.upsample(x)                 # (B, 16, 64, 64)

        x = F.relu(self.conv4(x))            # (B, 32, 64, 64)
        x = self.upsample(x)                 # (B, 32, 128, 128)

        x = F.relu(self.conv3(x))            # (B, 32, 128, 128)
        x = F.relu(self.conv2(x))            # (B, 32, 128, 128)
        x = self.conv1(x)                    # (B, 3, 128, 128)

        return x
    
class Autoencoder2(nn.Module):
    def __init__(self, latent_dim=1411):
        super().__init__()
        self.encoder = Encoder2(latent_dim)
        self.decoder = Decoder2(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out



#################################
class Encoder3(nn.Module):
    def __init__(self, latent_dim=1674):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        x = F.relu(self.conv2(x))   
        x = self.pool(x)
                
        x = self.flatten(x)        
        x = self.fc(x)            
        return x
    
class Decoder3(nn.Module):
    def __init__(self, latent_dim=1674):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 8 * 32 * 32)

        self.conv3 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       
        x = x.view(-1, 16, 16, 16)          

        x = F.relu(self.conv3(x))            
        x = self.upsample(x)                 

        x = F.relu(self.conv2(x))            
        x = self.conv1(x) 
        
        return x
    
class Autoencoder3(nn.Module):
    def __init__(self, latent_dim=1674):
        super().__init__()
        self.encoder = Encoder3(latent_dim)
        self.decoder = Decoder3(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

#################################
class Encoder4(nn.Module):
    def __init__(self, latent_dim=562):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 16, kernel_size=3, padding=1)


        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 4 * 16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))  
        
        x = F.relu(self.conv3(x)) 
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x
    
class Decoder4(nn.Module):
    def __init__(self, latent_dim=562):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 4 * 4 * 16)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       
        x = x.view(-1, 16, 4, 4) 
        
        x = F.relu(self.conv7(x))         
        x = self.upsample(x)
        x = F.relu(self.conv6(x))
        x = self.upsample(x)
        x = F.relu(self.conv5(x))
        x = self.upsample(x)
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))                  
        x = F.relu(self.conv2(x))            
        x = self.upsample(x)        
        x = self.conv1(x) 
        
        return x
    
class Autoencoder4(nn.Module):
    def __init__(self, latent_dim=562):
        super().__init__()
        self.encoder = Encoder4(latent_dim)
        self.decoder = Decoder4(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

###############################

class Encoder5(nn.Module):
    def __init__(self, latent_dim=685):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))  
        x = self.pool(x)
        
        x = F.relu(self.conv3(x)) 
        x = self.pool(x)
        
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x
    
class Decoder5(nn.Module):
    def __init__(self, latent_dim=685):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 16 * 16 * 8)

        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       
        x = x.view(-1, 8, 16, 16) 
        
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))   
        x = self.upsample(x)               
        x = F.relu(self.conv2(x))            
        x = self.upsample(x)        
        x = self.conv1(x) 
        
        return x
    
class Autoencoder5(nn.Module):
    def __init__(self, latent_dim=685):
        super().__init__()
        self.encoder = Encoder5(latent_dim)
        self.decoder = Decoder5(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
################################################

class Encoder6(nn.Module):
    def __init__(self, latent_dim=1262):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 64, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))  
        x = self.pool(x)
                
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x
    
class Decoder6(nn.Module):
    def __init__(self, latent_dim=1262):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 32 * 32 * 64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       
        x = x.view(-1, 64, 32, 32) 
        
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))   
        x = self.upsample(x)               
        x = self.conv1(x) 
        
        return x
    
class Autoencoder6(nn.Module):
    def __init__(self, latent_dim=1262):
        super().__init__()
        self.encoder = Encoder6(latent_dim)
        self.decoder = Decoder6(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
################################################

class Encoder7(nn.Module):
    def __init__(self, latent_dim=1960):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 64 * 16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = F.relu(self.conv3(x))
        x = self.pool(x)                
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x
    
class Decoder7(nn.Module):
    def __init__(self, latent_dim=1960):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 64 * 64 * 16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       
        x = x.view(-1, 64, 32, 32) 
        
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))   
        x = self.conv1(x) 
        
        return x
    
class Autoencoder7(nn.Module):
    def __init__(self, latent_dim=1960):
        super().__init__()
        self.encoder = Encoder7(latent_dim)
        self.decoder = Decoder7(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


##########################################
class Encoder8(nn.Module):
    def __init__(self, latent_dim=838):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        x = F.relu(self.conv2(x))  
        x = self.pool(x)                
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x
    
class Decoder8(nn.Module):
    def __init__(self, latent_dim=838):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 32 * 32 * 128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       
        x = x.view(-1, 128, 32, 32) 
        
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)   
        x = self.conv1(x) 
        
        return x
    
class Autoencoder8(nn.Module):
    def __init__(self, latent_dim=838):
        super().__init__()
        self.encoder = Encoder8(latent_dim)
        self.decoder = Decoder8(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
##########################################
class Encoder9(nn.Module):
    def __init__(self, latent_dim=148):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 4 * 32, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        x = F.relu(self.conv2(x))  
        x = self.pool(x)  
        x = F.relu(self.conv3(x))  
        x = self.pool(x)      
        x = F.relu(self.conv4(x))  
        x = self.pool(x)     
        x = F.relu(self.conv5(x))  
        x = self.pool(x)     
                        
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x
    
class Decoder9(nn.Module):
    def __init__(self, latent_dim=148):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 4 * 4 * 32)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       
        x = x.view(-1, 32, 4, 4) 
        
        x = F.relu(self.conv6(x))
        x = self.upsample(x)
        x = F.relu(self.conv5(x))
        x = self.upsample(x)   
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv1(x) 
        
        return x
    
class Autoencoder9(nn.Module):
    def __init__(self, latent_dim=148):
        super().__init__()
        self.encoder = Encoder9(latent_dim)
        self.decoder = Decoder9(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out