from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from .transform import return_transform

class CustomImageDataset(Dataset): 
    def __init__(self, csv, transform=True, autoencoder=True, data_per_class=None): 
        self.df = pd.read_csv(csv) 

        if data_per_class:
            self.df = pd.concat([self.df[self.df['class'] == 0].head(data_per_class), self.df[self.df['class'] == 1 ].head(data_per_class)])

        self.transform = return_transform() if transform else None
        self.autoencoder = autoencoder 

    def __len__(self): 
        return len(self.df) 

    def __getitem__(self, idx): 
        img_path = self.df.iloc[idx, 0] 
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            
        label = self.df.iloc[idx, 1] 

        if self.transform: 
            image = self.transform(image) 

        if self.autoencoder: 
            label = image 

        return np.array(image), label, idx