import os
import random
import transform
from transform import CATnDOG, transform
from torch.utils.data import DataLoader

img_file = os.listdir("data/train")

img_file = list(filter(lambda x: x != 'train', img_file))

def train_path(p):
    return f"data/train/{p}"

img_file = list(map(train_path, img_file))

# print("Train Images", len(img_file))

random.shuffle(img_file)

train = img_file[:20000]
test = img_file[20000:]

# print(len(train), len(test))

##traindataset
train_ds = CATnDOG(train, transform)
train_dl = DataLoader(train_ds, batch_size=100)
# print(len(train_ds), len(train_dl))
##testdataset
test_ds = CATnDOG(test, transform)
test_dl = DataLoader(test_ds, batch_size=100)
# print(len(test_ds), len(test_dl))

