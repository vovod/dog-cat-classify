import os
import torch
from model import CATnDOGconv
from transform import testCATnDOG, transform
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

test_file = os.listdir("data/test/")

test_file = list(filter(lambda x: x != 'test', test_file))

path_pt = "weights/train_epoch8.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CATnDOGconv().to(device)
model.load_state_dict(torch.load(path_pt))
model.eval()


def test_path(p):
    return f"data/test/{p}"

test_file = list(map(test_path, test_file))

test_ds = testCATnDOG(test_file, transform)
test_dl = DataLoader(test_ds, batch_size=100)

# print(len(test_dl))

dog_probs = []

with torch.no_grad():
    for X, fileid in tqdm(test_dl):
        X = X.to(device)
        preds = model(X)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(fileid), preds_list))

for img, probs in zip(test_file[:20], dog_probs[:20]):
    pil_im = Image.open(img, 'r')
    label = "dog" if probs[1] > 0.5 else "cat"
    title = "prob of dog: " + str(probs[1]) + " Classified as: " + label
    plt.figure()
    plt.imshow(pil_im)
    plt.suptitle(title)
    plt.show()
        