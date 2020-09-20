import torch 
import torch.optim as optim 
from torch.utils.data import DataLoader

import config 
from dataset import CloudDataset
from engine import train_fn, test_fn
from model import UNet
from loss import BCEDiceLoss


def run_training():
    train_dataset = CloudDataset(

    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size = config.BATCH_SIZE, 
        num_workers = cofig.NUM_WORKERS, 
        shuffle=True,
        pin_memory= config.PIN_MEMORY
    )
    test_dataset = CloudDataset(

    )
    test_loader = DataLoader(
        train_dataset, 
        batch_size = config.TEST_BATCH_SIZE, 
        num_workers = cofig.NUM_WORKERS, 
        shuffle=False,
        pin_memory= config.PIN_MEMORY
    )
    
    model = UNet()
    model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR, 
                            weight_decay=config.WEIGHT_DECAY)
    loss_fn = BCEDiceLoss()
    for epoch in range(config.EPOCHS):
        train_loss = train_fn()
        eval_loss, eval_accuracy = test_fn(model, test_loader, loss_fn)
        print(
           f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={eval_loss}, Test Accuracy={eval_accuracy}" 
        )
    print('Training_Completed')

if __name__ == "__main__":
    run_training