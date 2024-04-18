import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from torch.utils.data import Dataset
import cv2

def load_small_data(dir):

    data = np.load(dir)
    lst = data.files
    X = data[lst[0]]
    y = data[lst[1]]

    return X,y

def resize(width, height, images):
    output = []

    for image in images: 
        output.append(cv2.resize(image, (width, height)))
    return output

class JerseyDataset(Dataset):

    def __init__(self, images, labels, transform=None):
        
        """
        img_dir: the directory with the images
        transform: desired transformations
        target_transform:
        
        """

        self.labels = labels
        self.transform = transform

        self.W = images.shape[-1]
        self.H = images.shape[-2]
        self.channels = images.shape[1]

        self.images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        label = self.labels[idx]

        image = self.transform(image.reshape(-1,self.channels)[:,np.newaxis,:])

        return image.view(self.channels, self.H, self.W), label