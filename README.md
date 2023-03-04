# DOG vs CAT classify
**My project to classify cat and dog using pytorch!**  
  
![DOG](https://i.ibb.co/2h8M1p8/Figure-2.png)
![CAT](https://i.ibb.co/SJ1C3rK/Figure-1.png)
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
### Preprocess data:
```
preprocess.py
```
### Training with custom model:
Before training, create a folder name **weights** to save model.  
```
train.py
```
### And predicts:
```
predict.py
```
## This is the end!
##### The model from [Juan Zamora](https://hashnode.com/@doczamora).  
#### Thank you for stopping by!
