from torch import nn


import torch.nn.functional as F

from src.config.config import Config

LATENT_DIMS = Config.LATENT_DIMS

class SkipEncoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 16, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))   # (B, 8, 128, 128)
        x = self.pool(x1)            # (B, 8, 64, 64)

        x2 = F.relu(self.conv2(x))   # (B, 16, 64, 64)
        x = self.pool(x2)            # (B, 16, 32, 32)

        x = self.flatten(x)         # (B, 16384)
        z = self.fc(x)              # (B, LATENT_DIMS[0])
        return z, x1, x2
    
class SkipDecoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 32 * 32 * 16)

        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, skip1, skip2):
        x = self.fc(z)                       # (B, 16384)
        x = x.view(-1, 16, 32, 32)           # (B, 16, 32, 32)

        x = F.relu(self.conv1(x))            # (B, 16, 32, 32)
        x = self.upsample(x)                 # (B, 16, 64, 64)
        x = x + skip2                        # Skip connection

        x = F.relu(self.conv2(x))            # (B, 8, 64, 64)
        x = self.upsample(x)                 # (B, 8, 128, 128)
        x = x + skip1                        # Skip connection

        x = self.conv3(x)                    # (B, 3, 128, 128)
        return x
    
class SkipAutoencoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder0(latent_dim)
        self.decoder = SkipDecoder0(latent_dim)

    def forward(self, x):
        z, skip1, skip2 = self.encoder(x)
        out = self.decoder(z, skip1, skip2)
        return out

#################################
class SkipEncoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 32, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))   # (B, 32, 128, 128)
        x = self.pool(x1)            # (B, 32, 64, 64)

        x2 = F.relu(self.conv2(x))   # (B, 8, 64, 64)
        x = self.pool(x2)            # (B, 8, 32, 32)

        x3 = F.relu(self.conv3(x))   # (B, 64, 32, 32)
        x = self.pool(x3)           # (B, 64, 16, 16)
        
        x4 = F.relu(self.conv4(x))   # (B, 32, 16, 16)
        
        x = self.flatten(x4)         # (B, 8192)
        z = self.fc(x)              # (B, LATENT_DIMS[1])
        return z, x1, x2, x3, x4

class SkipDecoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 16 * 16 * 32)

        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, z, x1, x2, x3, x4):
        x = self.fc(z)                       # (B, 8192)
        x = x.view(-1, 32, 16, 16)           # (B, 32, 16, 16)

        x = F.relu(self.conv1(x))            # (B, 32, 16, 16)
        x = x + x4                        # Skip connection
        x = F.relu(self.conv2(x))            # (B, 64, 16, 16)
        x = self.upsample(x)                 # (B, 64, 32, 32)
        x = x + x3                        # Skip connection

        x = F.relu(self.conv3(x))            # (B, 8, 32, 32)
        x = self.upsample(x)                 # (B, 8, 64, 64)
        x = x + x2                        # Skip connection
        x = F.relu(self.conv4(x))            # (B, 32, 64, 64)
        x = self.upsample(x)                 # (B, 32, 128, 128)
        x = x + x1                        # Skip connection
        x = self.conv5(x)                    # (B, 3, 128, 128)

        return x
    
class SkipAutoencoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder1(latent_dim)
        self.decoder = SkipDecoder1(latent_dim)

    def forward(self, x):
        z, x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(z, x1, x2, x3, x4)
        return out


#################################
class SkipEncoder2(nn.Module):
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
        self.fc = nn.Linear(16 * 16 * 16, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))   # (B, 8, 128, 128)

        x2 = F.relu(self.conv2(x1))   # (B, 16, 64, 64)

        x3 = F.relu(self.conv3(x2))   # (B, 16, 64, 64)
        x = self.pool(x3)            # (B, 16, 32, 32)

        x4 = F.relu(self.conv4(x))   # (B, 16, 64, 64)
        x = self.pool(x4)            # (B, 16, 32, 32)
        
        x5 = F.relu(self.conv5(x))   # (B, 16, 64, 64)
        x = self.pool(x5)            # (B, 16, 32, 32)

        x = self.flatten(x)         # (B, 16384)
        x = self.fc(x)              # (B, 1411)
        return x, x1, x2, x3, x4, x5

class SkipDecoder2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[2]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 16 * 16 * 16)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, x1, x2, x3, x4, x5):
        x = self.fc(z)                       # (B, 4096)
        x = x.view(-1, 16, 16, 16)           # (B, 16, 16, 16)

        x = F.relu(self.conv6(x))            # (B, 16, 16, 16)
        x = self.upsample(x)                 # (B, 16, 32, 32)
        x = x + x5                        # Skip connection

        x = F.relu(self.conv5(x))            # (B, 16, 32, 32)
        x = self.upsample(x)                 # (B, 16, 64, 64)
        x = x + x4                        # Skip connection
        
        x = F.relu(self.conv4(x))            # (B, 32, 64, 64)
        x = self.upsample(x)                 # (B, 32, 128, 128)
        x = x + x3                        # Skip connection
        
        x = F.relu(self.conv3(x))            # (B, 32, 128, 128)
        x = x + x2                        # Skip connection
        
        x = F.relu(self.conv2(x))            # (B, 32, 128, 128)
        x = x + x1                        # Skip connection
        
        x = self.conv1(x)                    # (B, 3, 128, 128)

        return x
    
class SkipAutoencoder2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[2]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder2(latent_dim)
        self.decoder = SkipDecoder2(latent_dim)

    def forward(self, x):
        z, x1, x2, x3, x4, x5 = self.encoder(x)
        out = self.decoder(z, x1, x2, x3, x4, x5)
        return out



#################################
class SkipEncoder3(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[3]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 8, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  
        x = self.pool(x1)
        x2 = F.relu(self.conv2(x))   
        x = self.pool(x2)
                
        x = self.flatten(x)        
        x = self.fc(x)            
        return x, x1, x2
    
class SkipDecoder3(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[3]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 8 * 32 * 32)

        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, x1, x2):
        x = self.fc(x)                       
        x = x.view(-1, 8, 32, 32)          

        x = F.relu(self.conv3(x))            
        x = self.upsample(x)        
        x = x + x2                        # Skip connection         

        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = x + x1                        # Skip connection
                    
        x = self.conv1(x) 
        
        return x
    
class SkipAutoencoder3(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[3]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder3(latent_dim)
        self.decoder = SkipDecoder3(latent_dim)

    def forward(self, x):
        z, x1, x2 = self.encoder(x)
        out = self.decoder(z, x1, x2)
        return out

#################################
class SkipEncoder4(nn.Module):
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
        self.fc = nn.Linear(4 * 4 * 16, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  
        x = self.pool(x1)
        
        x2 = F.relu(self.conv2(x))  
        
        x3 = F.relu(self.conv3(x2)) 
        x = self.pool(x3)
        
        x4 = F.relu(self.conv4(x))
        x = self.pool(x4)
        
        x5 = F.relu(self.conv5(x))
        x = self.pool(x5)
        
        x6 = F.relu(self.conv6(x))
        x = self.pool(x6)
        
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x, x1, x2, x3, x4, x5, x6
    
class SkipDecoder4(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[4]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 4 * 4 * 16)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, skip1, skip2, skip3, skip4, skip5, skip6):
        x = self.fc(x)                       
        x = x.view(-1, 16, 4, 4) 
        
        x = F.relu(self.conv7(x))         
        x = self.upsample(x)
        x = x + skip6                        # Skip connection
        
        x = F.relu(self.conv6(x))
        x = self.upsample(x)
        x = x + skip5                        # Skip connection
        
        x = F.relu(self.conv5(x))
        x = self.upsample(x)
        x = x + skip4                        # Skip connection
        
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = x + skip3                        # Skip connection
        
        x = F.relu(self.conv3(x))                  
        x = x + skip2                        # Skip connection
        
        x = F.relu(self.conv2(x))            
        x = self.upsample(x)        
        x = x + skip1                        # Skip connection

                
        x = self.conv1(x) 
        
        return x
    
class SkipAutoencoder4(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[4]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder4(latent_dim)
        self.decoder = SkipDecoder4(latent_dim)

    def forward(self, x):
        z, skip1, skip2, skip3, skip4, skip5, skip6 = self.encoder(x)
        out = self.decoder(z, skip1, skip2, skip3, skip4, skip5, skip6)
        return out

###############################

class SkipEncoder5(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[5]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 8, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  
        x = self.pool(x1)
        
        x2 = F.relu(self.conv2(x))  
        x = self.pool(x2)
        
        x3 = F.relu(self.conv3(x)) 
        x = self.pool(x3)
        
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x, x1, x2, x3
    
class SkipDecoder5(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[5]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 8 * 16 * 16)

        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, skip1, skip2, skip3):
        x = self.fc(z)                       
        x = x.view(-1, 8, 16, 16) 
        
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = x + skip3                        # Skip connection
        
        x = F.relu(self.conv3(x))   
        x = self.upsample(x)               
        x = x + skip2                        # Skip connection
        
        x = F.relu(self.conv2(x))            
        x = self.upsample(x)        
        x = x + skip1                        # Skip connection
        
        x = self.conv1(x) 
        
        return x
    
class SkipAutoencoder5(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[5]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder5(latent_dim)
        self.decoder = SkipDecoder5(latent_dim)

    def forward(self, x):
        z, skip1, skip2, skip3 = self.encoder(x)
        out = self.decoder(z, skip1, skip2, skip3)
        return out
    
################################################

class SkipEncoder6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[6]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 64, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  
        x = self.pool(x1)
        
        x2 = F.relu(self.conv2(x))  
        x = self.pool(x2)
                
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x, x1, x2
    
class SkipDecoder6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[6]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 32 * 32 * 64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, skip1, skip2):
        x = self.fc(z)                       
        x = x.view(-1, 64, 32, 32) 
        
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = x + skip2                        # Skip connection
        
        x = F.relu(self.conv2(x))   
        x = self.upsample(x)               
        x = x + skip1                        # Skip connection
        
        x = self.conv1(x) 
        
        return x
    
class SkipAutoencoder6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[6]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder6(latent_dim)
        self.decoder = SkipDecoder6(latent_dim)

    def forward(self, x):
        z, skip1, skip2 = self.encoder(x)
        out = self.decoder(z, skip1, skip2)
        return out
    
################################################

class SkipEncoder7(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[7]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 16, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  
        x = self.pool(x1)                
        x2 = F.relu(self.conv2(x))  
        x = self.pool(x2)
        x3 = F.relu(self.conv3(x))
        x = self.pool(x3)
        x = self.flatten(x)        
        
        z = self.fc(x)            
        return z, x1, x2, x3
    
class SkipDecoder7(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[7]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 16 * 16 * 16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, x1, x2, x3):
        x = self.fc(z)                       
        x = x.view(-1, 16, 16, 16) 
        
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = x + x3                        # Skip connection
        
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = x + x2                        # Skip connection
        
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = x + x1                        # Skip connection
        
        x = self.conv1(x) 
        
        return x
    
class SkipAutoencoder7(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[7]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder7(latent_dim)
        self.decoder = SkipDecoder7(latent_dim)

    def forward(self, x):
        z, x1, x2, x3 = self.encoder(x)
        out = self.decoder(z, x1, x2, x3)
        return out


##########################################
class SkipEncoder8(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[8]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 128, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  
        x = self.pool(x1)
        x2 = F.relu(self.conv2(x))  
        x = self.pool(x2)                
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x, x1, x2
    
class SkipDecoder8(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[8]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 32 * 32 * 128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, x1, x2):
        x = self.fc(x)                       
        x = x.view(-1, 128, 32, 32) 
        
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = x + x2                        # Skip connection
        
        x = F.relu(self.conv2(x))
        x = self.upsample(x) 
        x = x + x1                        # Skip connection
          
        x = self.conv1(x) 
        
        return x
    
class SkipAutoencoder8(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[8]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder8(latent_dim)
        self.decoder = SkipDecoder8(latent_dim)

    def forward(self, x):
        z, x1, x2 = self.encoder(x)
        out = self.decoder(z, x1, x2)
        return out
    
##########################################
class SkipEncoder9(nn.Module):
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
        self.fc = nn.Linear(4 * 4 * 32, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  
        x = self.pool(x1)
        x2 = F.relu(self.conv2(x))  
        x = self.pool(x2)  
        x3 = F.relu(self.conv3(x))  
        x = self.pool(x3)      
        x4 = F.relu(self.conv4(x))  
        x = self.pool(x4)     
        x5 = F.relu(self.conv5(x))  
        x = self.pool(x5)     
                        
        x = self.flatten(x)        
        
        x = self.fc(x)            
        return x, x1, x2, x3, x4, x5
    
class SkipDecoder9(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[9]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 4 * 4 * 32)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, skip1, skip2, skip3, skip4, skip5):
        x = self.fc(x)                       
        x = x.view(-1, 32, 4, 4) 
        
        x = F.relu(self.conv6(x))
        x = self.upsample(x)
        x += skip5                        # Skip connection
        
        x = F.relu(self.conv5(x))
        x = self.upsample(x)
        x += skip4                        # Skip connection
           
        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x += skip3                        # Skip connection
        
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x += skip2                        # Skip connection
        
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x += skip1                        # Skip connection
        
        x = self.conv1(x) 
        
        return x
    
class SkipAutoencoder9(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[9]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SkipEncoder9(latent_dim)
        self.decoder = SkipDecoder9(latent_dim)

    def forward(self, x):
        z, skip1, skip2, skip3, skip4, skip5 = self.encoder(x)
        out = self.decoder(z, skip1, skip2, skip3, skip4, skip5)
        return out