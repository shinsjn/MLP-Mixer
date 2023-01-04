import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms

trans = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor()])

trainset = torchvision.datasets.ImageFolder('C:/Users/shins/PycharmProjects/MLP-MIXER/data/dataset/Fast Food Classification V2/Train',transform = trans)
trainloader = DataLoader(trainset,batch_size = 4,shuffle = True)
dataiter = iter(trainloader)

