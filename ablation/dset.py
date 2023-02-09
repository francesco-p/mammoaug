import torch
import cv2
import os
import random
import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class BCDDataset:
    def __init__(self, root, df, extension="png", transform=None, return_path=False):

        # Extract pathhs and labels        
        self.path = np.array([os.path.join(root, f"{p}_{im}.{extension}") for p, im in zip(df["patient_id"].values, df["image_id"].values)])
        self.label = df["cancer"].values

        # Shuffle them accordingly and get back to list for paths
        self.path, self.label = unison_shuffled_copies(self.path, self.label)
        self.path = self.path.tolist()

        self.return_path = return_path
        self.transform = transform

    def __len__(self):
        return len(self.path)
    

    def __getitem__(self, index):
        image = cv2.imread(self.path[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        if image.shape == (512, 1024):
            image = np.moveaxis(image, [1], [0]) # .permute(1,0)

        #assert image.shape == (1024, 512), f'Image {self.path[index]} shape is {image.shape}'
        
        image = cv2.resize(image, (256,512))
        image = torch.tensor(image, dtype=torch.float32)
        

        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_path:
            return image, self.label[index], self.path[index]
        
        return image, self.label[index]




class AUG_BCDDataset:
    def __init__(self, root, root_malignant, root_benign, df, extension="png", transform=None, return_path=False):
        
        # Extract pathhs and labels        
        paths = [os.path.join(root, f"{p}_{im}.{extension}") for p, im in zip(df["patient_id"].values, df["image_id"].values)]
        
        # Get benign and malignant paths
        benign = [os.path.join(root_benign, x) for x in os.listdir(root_benign)]
        malignant = [os.path.join(root_malignant, x) for x in os.listdir(root_malignant)]
        
        # Concatenate
        self.path = np.array(paths + benign + malignant)
        self.label =np.hstack((df["cancer"].values, np.zeros(len(benign)), np.ones(len(malignant)))).astype(np.int64)
        # shuffle them accordingly and get back to list for paths
        self.path, self.label = unison_shuffled_copies(self.path, self.label)
        self.path = self.path.tolist()

        self.return_path = return_path
        self.transform = transform

    def __len__(self):
        return len(self.path)
    

    def __getitem__(self, index):
        image = cv2.imread(self.path[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        

        
        if image.shape == (512, 1024):
            image = np.moveaxis(image, [1], [0]) # .permute(1,0)

        #assert image.shape == (1024, 512), f'Image {self.path[index]} shape is {image.shape}'
        
        image = cv2.resize(image, (256,512))
        image = torch.tensor(image, dtype=torch.float32)
        

        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_path:
            return image, self.label[index], self.path[index]
        
        return image, self.label[index]
