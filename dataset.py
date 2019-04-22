import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms


# Read in some of the parameters we need.
with open('params.json') as json_file:  
    params = json.load(json_file)
    basepath = params["basepath"]
    train_csv = basepath "CSVs/allData.csv"

    
class ShapesDataset(Dataset):
    
    def __init__(self, transform=None, csv=train_csv):
        data_csv = pd.read_csv(csv, index_col=False)
        self.train_file_names = [name for name in data_csv["filename"]]
        
        self.nImages = len(self.train_file_names)
        images = np.zeros((self.nImages, 64, 64, 3), dtype=np.uint8)
        
        for n in range(self.nImages):
            images[n] = Image.open(self.train_file_names[n])
            
        self.transform = transform
        self.images = images
      
        
    def __getitem__(self, index):
        img = self.images[index]
        
        # Transform image to tensor.
        if self.transform is not None:
            img = self.transform(img)
            
        return img
       
        
    def __len__(self):
        return len(self.images)