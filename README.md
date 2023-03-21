# DOG vs CAT classify
**My project to classify cat and dog using pytorch!**  
  
![DOG](/Figure_2.png)
![CAT](/Figure_1.png)
## Requirements
CUDA, conda installed.
```
conda create --name torch-cuda
conda activate torch-cuda
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
## Dataset
KAGGLE: [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data).  
The train folder contains 25,000 images of dogs and cats.  
The test folder contains 12,500 images of dogs and cats - non classify.
## Let's start!
### Preprocessing data:
```
preprocess.py
```
### Training with custom model:
Before training, create a folder name **weights** to save model.  
```
train.py
```
### And predicting:
```
predict.py
```
## This is the end!
##### The model from [Juan Zamora](https://hashnode.com/@doczamora).  
#### Thank you for stopping by!
