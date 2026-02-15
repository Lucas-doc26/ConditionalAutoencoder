from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class CustomImageDataset(Dataset): 
    def __init__(self, csv, transform=None, autoencoder=True): 
        self.df = pd.read_csv(csv) 
        self.transform = transform 
        self.autoencoder = autoencoder 

    def __len__(self): 
        return len(self.df) 

    def __getitem__(self, idx): 
        img_path = self.df.iloc[idx, 0] 
        image = Image.open(img_path).convert("RGB") 
        label = self.df.iloc[idx, 1] 

        if self.transform: 
            image = self.transform(image) 

        if self.autoencoder: 
            label = image 

        return image, label, idx