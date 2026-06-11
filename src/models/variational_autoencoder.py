import torch.nn as nn
import torch.nn.functional as F
import torch
from src.config.config import Config

LATENT_DIMS = Config().LATENT_DIMS


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_hat, x, reduction="mean")

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    loss = recon + beta * kl

    return loss, recon.detach(), kl.detach()


#Latent dims: 1849, 467, 1411, 1674, 562, 685, 1262, 1960, 838, 1417
#Estavam muito grandes, por isso troquei tudo por valores aleatórios entre 256 e 512, para ver se melhora a qualidade das imagens reconstruídas.


class VariationalEncoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()

        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 16, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 8, 64, 64)
        x = self.pool(x)            # (B, 8, 32, 32)

        x = F.relu(self.conv2(x))   # (B, 16, 32, 32)
        x = self.pool(x)            # (B, 16, 16, 16)

        x = self.flatten(x)         # (B, 4096)

        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class VariationalDecoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 16 * 16 * 16)

        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)                       # (B, 4096)
        x = x.view(-1, 16, 16, 16)           # (B, 16, 16, 16)

        x = F.relu(self.conv1(x))            # (B, 16, 16, 16)
        x = self.upsample(x)                 # (B, 16, 32, 32)

        x = F.relu(self.conv2(x))            # (B, 8, 32, 32)
        x = self.upsample(x)                 # (B, 8, 64, 64)

        x = torch.sigmoid(self.conv3(x))              # (B, 3, 64, 64)

        return x
    
class VariationalAutoencoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder0(latent_dim)
        self.decoder = VariationalDecoder0(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar

class VariationalEncoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 8 * 32, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 32, 64, 64)
        x = self.pool(x)            # (B, 32, 32, 32)

        x = F.relu(self.conv2(x))   # (B, 8, 32, 32)
        x = self.pool(x)            # (B, 8, 16, 16)

        x = F.relu(self.conv3(x))   # (B, 64, 16, 16)
        x = self.pool(x)            # (B, 64, 8, 8)

        x = F.relu(self.conv4(x))   # (B, 32, 8, 8)

        x = self.flatten(x)         # (B, 2048)
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class VariationalDecoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, 8 * 8 * 32)

        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)                       # (B, 2048)
        x = x.view(-1, 32, 8, 8)            # (B, 32, 8, 8)

        x = F.relu(self.conv1(x))            # (B, 32, 8, 8)
        x = F.relu(self.conv2(x))            # (B, 64, 8, 8)
        x = self.upsample(x)                 # (B, 64, 16, 16)

        x = F.relu(self.conv3(x))            # (B, 8, 16, 16)
        x = self.upsample(x)                 # (B, 8, 32, 32)

        x = F.relu(self.conv4(x))            # (B, 32, 32, 32)
        x = self.upsample(x)                 # (B, 32, 64, 64)

        x = self.conv5(x)                    # (B, 3, 64, 64)

        return x
    
class VariationalAutoencoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder1(self.latent_dim)
        self.decoder = VariationalDecoder1(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar


#################################
class VariationalEncoder2(nn.Module):
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
        self.fc = nn.Linear(8 * 8 * 16, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 32, 64, 64)

        x = F.relu(self.conv2(x))   # (B, 32, 64, 64)

        x = F.relu(self.conv3(x))   # (B, 16, 64, 64)
        x = self.pool(x)            # (B, 16, 32, 32)

        x = F.relu(self.conv4(x))   # (B, 16, 32, 32)
        x = self.pool(x)            # (B, 16, 16, 16)

        x = F.relu(self.conv5(x))   # (B, 16, 16, 16)
        x = self.pool(x)            # (B, 16, 8, 8)

        x = self.flatten(x)         # (B, 1024)
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class VariationalDecoder2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[2]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 8 * 8 * 16)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)                       # (B, 1024)
        x = x.view(-1, 16, 8, 8)            # (B, 16, 8, 8)

        x = F.relu(self.conv6(x))            # (B, 16, 8, 8)
        x = self.upsample(x)                 # (B, 16, 16, 16)

        x = F.relu(self.conv5(x))            # (B, 16, 16, 16)
        x = self.upsample(x)                 # (B, 16, 32, 32)

        x = F.relu(self.conv4(x))            # (B, 16, 32, 32)
        x = self.upsample(x)                 # (B, 16, 64, 64)

        x = F.relu(self.conv3(x))            # (B, 32, 64, 64)
        x = F.relu(self.conv2(x))            # (B, 32, 64, 64)
        x = self.conv1(x)                    # (B, 3, 64, 64)

        return x
    
class VariationalAutoencoder2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[2]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder2(self.latent_dim)
        self.decoder = VariationalDecoder2(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar



#################################
class VariationalEncoder3(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[3]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 16 * 16, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        x = F.relu(self.conv2(x))   
        x = self.pool(x)
                
        x = self.flatten(x)        
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class VariationalDecoder3(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[3]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 8 * 16 * 16)

        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 8, 16, 16)

        x = F.relu(self.conv3(x))
        x = self.upsample(x)                 # 16 -> 32

        x = F.relu(self.conv2(x))
        x = self.upsample(x)                 # 32 -> 64
        x = torch.sigmoid(self.conv1(x))
        
        return x
    
class VariationalAutoencoder3(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[3]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder3(self.latent_dim)
        self.decoder = VariationalDecoder3(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar

#################################
class VariationalEncoder4(nn.Module):
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
        self.fc = nn.Linear(16 * 2 * 2, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

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
        
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class VariationalDecoder4(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[4]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 2 * 2 * 16)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 16, 2, 2)

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

class VariationalAutoencoder4(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[4]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder4(self.latent_dim)
        self.decoder = VariationalDecoder4(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar

###############################

class VariationalEncoder5(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[5]):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 8 * 8, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))  
        x = self.pool(x)
        
        x = F.relu(self.conv3(x)) 
        x = self.pool(x)
        
        x = self.flatten(x)        
        
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class VariationalDecoder5(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[5]):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, 8 * 8 * 8)

        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 8, 8, 8)

        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv1(x)

        return x

class VariationalAutoencoder5(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[5]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder5(self.latent_dim)
        self.decoder = VariationalDecoder5(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar
    
################################################

class VariationalEncoder6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[6]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 64, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))  
        x = self.pool(x)
                
        x = self.flatten(x)        
        
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class VariationalDecoder6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[6]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 16 * 16 * 64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 16, 16)

        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv1(x)

        return x

class VariationalAutoencoder6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[6]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder6(self.latent_dim)
        self.decoder = VariationalDecoder6(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar
    
################################################

class VariationalEncoder7(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[7]):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 16, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = F.relu(self.conv3(x))
        x = self.pool(x)                
        x = self.flatten(x)        
        
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class VariationalDecoder7(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[7]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 32 * 32 * 16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 16, 32, 32)

        x = F.relu(self.conv4(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = self.conv1(x)

        return x

class VariationalAutoencoder7(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[7]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder7(self.latent_dim)
        self.decoder = VariationalDecoder7(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar


##########################################
class VariationalEncoder8(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[8]):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 128, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)
        x = F.relu(self.conv2(x))  
        x = self.pool(x)                
        x = self.flatten(x)        
        
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class VariationalDecoder8(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[8]):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, 16 * 16 * 128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 16, 16)

        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv1(x)

        return x

class VariationalAutoencoder8(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[8]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder8(self.latent_dim)
        self.decoder = VariationalDecoder8(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar
    
##########################################
class VariationalEncoder9(nn.Module):
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
        self.fc = nn.Linear(2 * 2 * 32, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, latent_dim)

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
        
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class VariationalDecoder9(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[9]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 2 * 2 * 32)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 2, 2)

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

class VariationalAutoencoder9(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[9]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder9(self.latent_dim)
        self.decoder = VariationalDecoder9(self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decoder(z)

        return x_hat, mu, logvar