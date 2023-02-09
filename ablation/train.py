from dset import BCDDataset, AUG_BCDDataset
import torch
import timm
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from utils import split_train_val_test, split_train_val
import torchvision
import torch.nn.functional as F
import random
import numpy as np

def set_seeds(seed):
    """ Set reproducibility seeds """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    SEED = 77
    set_seeds(SEED)
    DEVICE = "cuda"
    CSV_DATA_FILE = "/home/francesco/Documents/patch_titties/ablation/ablation.csv"
    OUTPUT_TRAIN_CSV_FILE = '/home/francesco/Documents/patch_titties/ablation/metrics_train.csv'
    OUTPUT_VAL_CSV_FILE = '/home/francesco/Documents/patch_titties/ablation/metrics_val.csv'
    ROOT = "/home/francesco/data/bcd2022/rectangular_dset/"
    ROOT_BENIGN = '/home/francesco/data/bcd2022/benign_augmented/'
    ROOT_MALIGNANT = '/home/francesco/data/bcd2022/malignant_augmented/'
    BATCH_SIZE = 32
    EPOCHS = 200
    MODEL = 'tf_efficientnetv2_s_in21ft1k'
    LR = 3.0e-4

    with open(OUTPUT_TRAIN_CSV_FILE, 'w') as f:
        f.write('epoch,loss,acc\n')
    with open(OUTPUT_VAL_CSV_FILE, 'w') as f:
        f.write('epoch,loss,acc\n')
    
    VALIDATION_EXAMPLES = 100
    
    # Read csv and shuffle it
    df = pd.read_csv(CSV_DATA_FILE)
    df = df.sample(frac=1)

    # Extract validation examples
    validation_df = df.iloc[:VALIDATION_EXAMPLES]
    val_loader = DataLoader(BCDDataset(ROOT, validation_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    

    # Extract TRAIN examples
    df = df.iloc[VALIDATION_EXAMPLES:]
    dataset_aug = AUG_BCDDataset(ROOT, ROOT_MALIGNANT, ROOT_BENIGN, df)
    #train_loader, val_loader = split_train_val(dataset, batch_size=BSIZE)
    train_loader = DataLoader(dataset_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


    # instantiate timm resnet18
    model = timm.create_model(MODEL, in_chans=1, pretrained=False, num_classes=2)
    model = model.to(DEVICE)
    model.train()

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):

        model.train()
        for x,y in train_loader:
            x = x.unsqueeze(1).to(DEVICE)
            y = y.to(DEVICE)
            #x = x[:,:,250:-250,50:-10]
            x = x[:,:,30:-30,:]
            y_onehot = F.one_hot(y, num_classes=2).to(torch.float32)

            y_hat = model(x)
            loss = loss_fn(y_hat, y_onehot)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            acc = (y_hat.argmax(dim=1) == y).sum() / len(y)
            print(f'{epoch},{loss.item()},{acc.item()}')
            with open(OUTPUT_TRAIN_CSV_FILE, 'a') as f:
                f.write(f'{epoch},{loss.item()},{acc.item()}\n')


        # Evaluate
        model.eval()
        with torch.no_grad():
            for x,y in val_loader:
                x = x.unsqueeze(1).to(DEVICE)
                y = y.to(DEVICE)
                #x = x[:,:,250:-250,50:-10]
                x = x[:,:,30:-30,:]
                y_onehot = F.one_hot(y, num_classes=2).to(torch.float32)

                y_hat = model(x)
                loss = loss_fn(y_hat, y_onehot)
                acc = (y_hat.argmax(dim=1) == y).sum() / len(y)
                print(f'    val - e:{epoch} l:{loss.item()} a:{acc.item()}')
                with open(OUTPUT_VAL_CSV_FILE, 'a') as f:
                    f.write(f'{epoch},{loss.item()},{acc.item()}\n')


if __name__ == '__main__':
    main()