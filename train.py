import time
import pandas as pd
import numpy as np
import torch
import os
import torchvision
import torch.nn as nn
from model import CATnDOGconv
from preprocess import train_dl, test_dl
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_dict = "weights/"

model = CATnDOGconv().to(device)

losses = []
accuracies = []
epoches = 10

start = time.time()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# print(device)

for epoch in range(epoches):
    epoch_loss = 0
    epoch_accuracy = 0

    for X, y in tqdm(train_dl):
        X = X.to(device)
        y = y.to(device)

        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = ((preds.argmax(dim=1)==y).float().mean())
        epoch_accuracy+=accuracy
        epoch_loss+=loss

        # print('.', end='', flush=True)
    
    epoch_accuracy = epoch_accuracy/len(train_dl)
    accuracies.append(epoch_accuracy)
    epoch_loss = epoch_loss/len(train_dl)
    losses.append(epoch_loss)

    print("\n --- Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(epoch, epoch_loss, epoch_accuracy, time.time() - start))

    with torch.no_grad():
        test_epoch_loss = 0
        test_epoch_accuracy = 0

        for test_X, test_y in tqdm(test_dl):
            test_X = test_X.to(device)
            test_y = test_y.to(device)

            test_preds = model(test_X)
            test_loss = loss_fn(test_preds, test_y)

            test_epoch_loss += test_loss
            test_accuracy = ((test_preds.argmax(dim=1)==test_y).float().mean())

            test_epoch_accuracy += test_accuracy

        test_epoch_accuracy = test_epoch_accuracy/len(test_dl)
        test_epoch_loss = test_epoch_loss / len(test_dl)

        print("Epoch: {}, test loss: {:.4f}, test acc: {:.4f}, time: {}\n".format(epoch, test_epoch_loss, test_epoch_accuracy, time.time() - start))
        torch.save(model.state_dict(), os.path.join(
            load_dict, "train_epoch{}.pth".format(epoch+1)))