import os 
import cv2 
import pandas as pd 
import numpy as np
import albumentations as albu
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_training_augmentation, get_validation_augmentation
from utils import make_mask




# Dataset class
class CloudDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame = None,
        datatype: str = "train",
        img_ids: np.array = None,
        transforms=albu.Compose([albu.HorizontalFlip()]), #, AT.ToTensor()
    ):
        self.df = df
        if datatype != "test":
            img_paths = '../data_resized/'
            self.data_folder = f"{img_paths}/train_images_525/train_images_525"
        else:
            img_paths = '../data_resized/'
            self.data_folder = f"{img_paths}/test_images_525/test_images_525"
        self.img_ids = img_ids
        self.transforms = transforms

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = np.transpose(augmented["image"], [2, 0, 1])
        mask = np.transpose(augmented["mask"], [2, 0, 1])
        return img, mask

    def __len__(self):
        return len(self.img_ids)


    

if __name__ == "__main__":
    path = '../data_resized/'
    img_paths = '../data_resized/'
    SEED = 42
    MODEL_NO = 0
    N_FOLDS = 5

    train = pd.read_csv(f"{path}/train.csv")
    train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

    sub = pd.read_csv(f"{path}/sample_submission.csv")
    sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])

    # split data
    id_mask_count = (
        train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"]
        .apply(lambda x: x.split("_")[0])
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "img_id", "Image_Label": "count"})
    )
    ids = id_mask_count["img_id"].values
    li = [
        [train_index, test_index]
        for train_index, test_index in StratifiedKFold(
            n_splits=N_FOLDS,
        ).split(ids, id_mask_count["count"])
    ]
    train_ids, valid_ids = ids[li[MODEL_NO][0]], ids[li[MODEL_NO][1]]
    test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values

    print(f"training set   {train_ids[:5]}.. with length {len(train_ids)}")
    print(f"validation set {valid_ids[:5]}.. with length {len(valid_ids)}")
    print(f"testing set    {test_ids[:5]}.. with length {len(test_ids)}")
    

    # define dataset and dataloader
    num_workers = 2
    bs = 8
    train_dataset = CloudDataset(
        df=train,
        datatype="train",
        img_ids=train_ids,
        transforms=get_training_augmentation(),
    )
    valid_dataset = CloudDataset(
        df=train,
        datatype="valid",
        img_ids=valid_ids,
        transforms=get_validation_augmentation(),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers
    )