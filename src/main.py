import os 
import cv2 
import numpy as np 
import pandas as pd 
from tqdm.auto import tqdm as tq
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

from model import UNet
from loss import BCEDiceLoss
from dataset import CloudDataset
from utils import get_training_augmentation, get_validation_augmentation, 
from utils import dice_no_threshold, dice, make_mask


if __name__ == "__main__":
    path = r'../data_resized'
    img_paths = r'../data_resized'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


    num_workers = 2
    bs = 2
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
    model = UNet(n_channels=3, n_classes=4).float()
    model.to(device)
    criterion = BCEDiceLoss(eps=1.0, activation=None)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2, 
                                                        patience=2, cooldown=2)

    n_epochs = 2
    train_loss_list = []
    valid_loss_list = []
    dice_score_list = []
    lr_rate_list = []
    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        dice_score = 0.0

        model.train()
        bar = tq(train_loader, postfix={"train_loss":0.0})
        for data, target in bar:
            # move tensors to GPU if CUDA is available
            data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            #print(loss)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            bar.set_postfix(ordered_dict={"train_loss":loss.item()})

        model.eval()
        #del data, target
        with torch.no_grad():
            bar = tq(valid_loader, postfix={"valid_loss":0.0, "dice_score":0.0})
            for data, target in bar:
                # move tensors to GPU if CUDA is available
                data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
                dice_cof = dice_no_threshold(output.cpu(), target.cpu()).item()
                dice_score +=  dice_cof * data.size(0)
                bar.set_postfix(ordered_dict={"valid_loss":loss.item(), "dice_score":dice_cof})
        
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        dice_score = dice_score/len(valid_loader.dataset)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        dice_score_list.append(dice_score)
        lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])
        
        print('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f} Dice Score: {:.6f}'.format(
            epoch, train_loss, valid_loss, dice_score))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            valid_loss_min = valid_loss
        
        scheduler.step(valid_loss)