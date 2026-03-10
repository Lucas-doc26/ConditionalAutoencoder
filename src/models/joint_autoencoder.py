import torch
from torch import nn


import torch.nn.functional as F

from src.config.config import Config

LATENT_DIMS = [512 for i in range(10)]




def combined_loss(recon_x, x):
    recon_loss = F.mse_loss(recon_x, x)
    return recon_loss


class JointEncoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 16, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 8, 128, 128)
        x = self.pool(x)            # (B, 8, 64, 64)

        x = F.relu(self.conv2(x))   # (B, 16, 64, 64)
        x = self.pool(x)            # (B, 16, 32, 32)

        x = self.flatten(x)         # (B, 16384)
        x = self.fc(x)              # (B, 1849)
        return x
    
class JointDecoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 32 * 32 * 16)

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
    
class JointAutoencoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder0(self.latent_dim)
        self.joint_decoder = JointDecoder0(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out

#################################
class JointEncoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 32, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 8, 128, 128)
        x = self.pool(x)            # (B, 8, 64, 64)

        x = F.relu(self.conv2(x))   # (B, 16, 64, 64)
        x = self.pool(x)            # (B, 16, 32, 32)

        x = F.relu(self.conv3(x))   # (B, 16, 64, 64)
        x = self.pool(x)           # (B, 16, 32, 32)
        
        x = F.relu(self.conv4(x))   # (B, 16, 64, 64)
        
        x = self.flatten(x)         # (B, 16384)
        x = self.fc(x)              # (B, LATENT_DIMS[1])
        return x
    
class JointDecoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, 16 * 16 * 32)

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
    
class JointAutoencoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder1(self.latent_dim)
        self.joint_decoder = JointDecoder1(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out


#################################
class JointEncoder2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[2]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 16, self.latent_dim)

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
    
class JointDecoder2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[2]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 16 * 16 * 16)

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
    
class JointAutoencoder2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[2]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder2(self.latent_dim)
        self.joint_decoder = JointDecoder2(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out



#################################
class JointEncoder3(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[3]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 32 * 32, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        x = F.relu(self.conv2(x))   
        x = self.pool(x)
                
        x = self.flatten(x)        
        x = self.fc(x)            
        return x
    
class JointDecoder3(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[3]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 8 * 32 * 32)

        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       
        x = x.view(-1, 8, 32, 32)          

        x = F.relu(self.conv3(x))            
        x = self.upsample(x)                 # 32 -> 64

        x = F.relu(self.conv2(x))            
        x = self.upsample(x)                 # 64 -> 128
        x = torch.sigmoid(self.conv1(x))
        
        return x
    
class JointAutoencoder3(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[3]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder3(self.latent_dim)
        self.joint_decoder = JointDecoder3(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out

#################################
class JointEncoder4(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[4]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 16, kernel_size=3, padding=1)


        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 4 * 4, self.latent_dim)

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
    
class JointDecoder4(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[4]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 4 * 4 * 16)

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
    
class JointAutoencoder4(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[4]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder4(self.latent_dim)
        self.joint_decoder = JointDecoder4(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out

###############################

class JointEncoder5(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[5]):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 16 * 16, self.latent_dim)

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
    
class JointDecoder5(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[5]):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, 16 * 16 * 8)

        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
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
    
class JointAutoencoder5(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[5]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder5(self.latent_dim)
        self.joint_decoder = JointDecoder5(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out
    
################################################

class JointEncoder6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[6]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 64, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))  
        x = self.pool(x)
                
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x
    
class JointDecoder6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[6]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 32 * 32 * 64)

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
    
class JointAutoencoder6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[6]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder6(self.latent_dim)
        self.joint_decoder = JointDecoder6(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out
    
################################################

class JointEncoder7(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[7]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 64 * 16, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = F.relu(self.conv3(x))
        x = self.pool(x)                
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x
    
class JointDecoder7(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[7]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 64 * 64 * 16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.fc(x)                       
        x = x.view(-1, 16, 64, 64) 
        
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))   
        x = self.conv1(x) 
        
        return x
    
class JointAutoencoder7(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[7]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder7(self.latent_dim)
        self.joint_decoder = JointDecoder7(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out


##########################################
class JointEncoder8(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[8]):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 128, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        x = F.relu(self.conv2(x))  
        x = self.pool(x)                
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x
    
class JointDecoder8(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[8]):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, 32 * 32 * 128)

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
    
class JointAutoencoder8(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[8]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder8(self.latent_dim)
        self.joint_decoder = JointDecoder8(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out
    
##########################################
class JointEncoder9(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[9]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 4 * 32, self.latent_dim)

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
    
class JointDecoder9(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[9]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 4 * 4 * 32)

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
    
class JointAutoencoder9(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[9]):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_encoder = JointEncoder9(self.latent_dim)
        self.joint_decoder = JointDecoder9(self.latent_dim)

    def forward(self, x):
        z = self.joint_encoder(x)
        out = self.joint_decoder(z)
        return out