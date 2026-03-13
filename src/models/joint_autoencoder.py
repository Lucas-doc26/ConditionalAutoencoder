from torch import nn
import torch.nn.functional as F

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
        self.fc = nn.Linear(32 * 32 * 16, latent_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))   # (B, 8, 128, 128)
        x = self.pool(x1)            # (B, 8, 64, 64)

        x2 = F.relu(self.conv2(x))   # (B, 16, 64, 64)
        x = self.pool(x2)            # (B, 16, 32, 32)

        x = self.flatten(x)         # (B, 16384)
        z = self.fc(x)              # (B, LATENT_DIMS[0])
        return z, x1, x2
    
class JointDecoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 32 * 32 * 16)

        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, Joint1, Joint2):
        x = self.fc(z)                       # (B, 16384)
        x = x.view(-1, 16, 32, 32)           # (B, 16, 32, 32)

        x = F.relu(self.conv1(x))            # (B, 16, 32, 32)
        x = self.upsample(x)                 # (B, 16, 64, 64)
        x = x + Joint2                        # Joint connection

        x = F.relu(self.conv2(x))            # (B, 8, 64, 64)
        x = self.upsample(x)                 # (B, 8, 128, 128)
        x = x + Joint1                        # Joint connection

        x = self.conv3(x)                    # (B, 3, 128, 128)
        return x
    
class JointAutoencoder0(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[0]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = JointEncoder0(latent_dim)
        self.decoder = JointDecoder0(latent_dim)

    def forward(self, x):
        z, Joint1, Joint2 = self.encoder(x)
        out = self.decoder(z, Joint1, Joint2)
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

class JointDecoder1(nn.Module):
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
        x = x + x4                        # Joint connection
        x = F.relu(self.conv2(x))            # (B, 64, 16, 16)
        x = self.upsample(x)                 # (B, 64, 32, 32)
        x = x + x3                        # Joint connection

        x = F.relu(self.conv3(x))            # (B, 8, 32, 32)
        x = self.upsample(x)                 # (B, 8, 64, 64)
        x = x + x2                        # Joint connection
        x = F.relu(self.conv4(x))            # (B, 32, 64, 64)
        x = self.upsample(x)                 # (B, 32, 128, 128)
        x = x + x1                        # Joint connection
        x = self.conv5(x)                    # (B, 3, 128, 128)

        return x
    
class JointAutoencoder1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMS[1]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = JointEncoder1(latent_dim)
        self.decoder = JointDecoder1(latent_dim)

    def forward(self, x):
        z, x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(z, x1, x2, x3, x4)
        return out