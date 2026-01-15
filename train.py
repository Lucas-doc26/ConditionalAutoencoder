import torch
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os
import pandas as pd 
import matplotlib.pyplot as plt
from autoencoders import *


class CustomImageDataset(Dataset):
    """Dataset customizado para imagens"""
    
    def __init__(self, csv, transform=None, autoencoder=True):
        self.df = pd.read_csv(csv)
        self.transform = transform
        self.autoencoder = autoencoder
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)

        if self.autoencoder == True:
            label = image
        
        return image, label


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormaliza imagem para visualização"""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    image = image * std + mean
    return torch.clamp(image, 0, 1)


def plot_reconstruction(original, reconstructed, model_name, dataset_name, save_path='./results/'):
    """Plot com 8 imagens de teste: original vs reconstruído"""
    
    # Criar diretório se não existir
    os.makedirs(save_path, exist_ok=True)
    
    # Denormalizar imagens
    original = denormalize(original.cpu())
    reconstructed = denormalize(reconstructed.detach().cpu())
    
    # Criar figura com 2 linhas e 8 colunas
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    fig.suptitle(f'{model_name} - {dataset_name} - Reconstruction Test', fontsize=16, fontweight='bold')
    
    for i in range(8):
        # Original
        orig_img = original[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstruído
        recon_img = reconstructed[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f'Reconstructed {i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Salvar figura
    filename = os.path.join(save_path, f'{model_name}_{dataset_name}_reconstruction.png')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"✓ Plot salvo: {filename}")
    plt.close()


transform = transforms.Compose([
    transforms.Resize((128, 128)),         
    transforms.ToTensor(),                   # HWC [0,255] -> CHW [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],           # ImageNet mean
        std=[0.229, 0.224, 0.225]              # ImageNet std
    )
])

for train in ['CNR', 'PKLot']:
    train_dataset = CustomImageDataset(csv=f'/home/lucas.ocunha/ConditionalAutoencoder/CSV/{train}/{train}_autoencoder_train.csv', transform=transform, autoencoder=True)
    val_dataset = CustomImageDataset(csv=f'/home/lucas.ocunha/ConditionalAutoencoder/CSV/{train}/{train}_autoencoder_validation.csv', transform=transform, autoencoder=True)
    test_dataset = CustomImageDataset(csv=f'/home/lucas.ocunha/ConditionalAutoencoder/CSV/{train}/{train}_autoencoder_test.csv', transform=transform, autoencoder=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for model_class in [Autoencoder]:
        model_name = model_class.__name__
        print(f"\n{'='*70}")
        print(f"Treinando {model_name} com dataset {train}")
        print(f"{'='*70}")

        model = model_class().to('cuda')
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 500
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Treino
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images = images.to('cuda')
                labels = labels.to('cuda')
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to('cuda')
                    labels = labels.to('cuda')
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        print(f"✓ Treinamento concluído!")
        
        # Salvar modelo
        os.makedirs('./models', exist_ok=True)
        torch.save(model.state_dict(), f'./models/{model_name}_{train}.pth')
        print(f"✓ Modelo salvo: ./models/{model_name}_{train}.pth")
        
        # Teste e Plot - Pegar primeiras 8 imagens
        model.eval()
        test_images, test_labels = None, None
        with torch.no_grad():
            for images, labels in test_loader:
                test_images = images[:8].to('cuda')
                test_labels = labels[:8].to('cuda')
                break
        
        # Predição
        with torch.no_grad():
            reconstructed = model(test_images)
        
        # Plot
        plot_reconstruction(test_images, reconstructed, model_name, train, save_path='./results/')
        
        print(f"✓ {model_name} - {train} finalizado!\n")
