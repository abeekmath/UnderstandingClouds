import os 
import cv2
import numpy as np
import pandas as pd 
import albumentations as alb 
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_train_augmentation, get_val_augmentation, make_mask


class CloudDataset(Dataset):
    def __init__(self, df=None, datatype='train', transforms=None):
        self.df = df
        if datatype != 'test':
            self.datafolder = '../data_resized/train_images_525/train_images_525'
        else:
            self.datafolder = '../data_resized/test_images_525/test_images_525'
        self.img_ids = img_ids
        self.transforms = transforms

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        img_path = os.path.join(self.datafolder, image_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = np.transpose(
            augmented["image"], [2, 0, 1], 
        )
        mask = np.transpose(
            augmented["mask"], [2, 0, 1], 
        )
        return img, mask

    def __len__(self):
        return len(self.img_ids)