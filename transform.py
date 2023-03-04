from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CATnDOG(Dataset):
    def __init__(self, image_path, transform):
        super().__init__()
        self.path = image_path
        self.len = len(self.path)
        self.transform = transform

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        path = self.path[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = 0 if 'cat' in path else 1
        return (image,label)
    
class testCATnDOG(Dataset):
    def __init__(self, image_path, transform):
        super().__init__()
        self.path = image_path
        self.len = len(self.path)
        self.transform = transform

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        path = self.path[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        fileid = path.split('/')[-1].split('.')[0]
        return (image, fileid)
