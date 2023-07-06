'''This is an implemention for the paper of ZfNet. You can find the paper under 
the link "https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf"'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchsummary import summary
import os
from zipfile import ZipFile
import urllib.request
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from collections import OrderedDict
from classes import class_names
#from utils import *

from collections import OrderedDict
import torch

# def load_model_with_diff_keys(pretrained_model_file, target_model):
#     """
#         Loads a pretrained model parameters into our target model.
#         We assume that there is only mistmatch between key names
#         and tensor dimension are same.
        
#         Parameters:
#         -pretrained_model_file: .pth file.
#         -target_model: model to load parameters.
        
#     """
#     pretrained_model = torch.load(pretrained_model_file)
#     new_state_dict = OrderedDict()
#     model_key = list(target_model.state_dict().keys())
    
#     count = 0
#     for key, value in pretrained_model.items():
#         new_key = model_key[count]
#         new_state_dict[new_key] = value
#         count += 1
        
#     target_model.load_state_dict(new_state_dict)
    
def load_model(pretrained_model_file, target_model):
    """
        Loads a pretrained model parameters into our target model.
        where target model parameters names are different from the
        pretrained one.
        
        Load parameters to deconv part too.
        
        Parameters:
        -pretrained_model_file: .pth file.
        -target_model: model to load parameters.
        
    """
    pretrained_model = torch.load(pretrained_model_file)
    new_state_dict_1 = OrderedDict()
    model_key = list(target_model.state_dict().keys())
    

    count = 0
    for key, value in pretrained_model.items():
        new_key = model_key[count]
        new_state_dict_1[new_key] = value
        count += 1

    mapping = {'features.conv1.weight': 'deconv_conv1.weight',
               'features.conv2.weight': 'deconv_conv2.weight',
               'features.conv3.weight': 'deconv_conv3.weight',
               'features.conv4.weight': 'deconv_conv4.weight',
               'features.conv5.weight': 'deconv_conv5.weight'}
    
    new_state_dict_2 = OrderedDict()
    # Load Deconv part
    for key, value in new_state_dict_1.items():
        if key in mapping:
            new_state_dict_2[mapping[key]] = value
    
    
    target_model.load_state_dict({**new_state_dict_1, **new_state_dict_2})


class ZFNet(nn.Module):
    
    def __init__(self):
        super(ZFNet, self).__init__()
        
        # CONV PART.
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1)),
            ('act1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)),
            ('conv2', nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0)),
            ('act2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)),
            ('conv3', nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)),
            ('act3', nn.ReLU()),
            ('conv4', nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)),
            ('act4', nn.ReLU()),
            ('conv5', nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)),
            ('act5', nn.ReLU()),
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True))
        ]))
    
        self.feature_outputs = [0]*len(self.features)
        self.switch_indices = dict()
        self.sizes = dict()


        self.classifier = nn.Sequential(OrderedDict([
            ('fc6', nn.Linear(9216, 4096)),
            ('act6', nn.ReLU()),
            ('fc7', nn.Linear(4096, 4096)),
            ('act7', nn.ReLU()),
            ('fc8', nn.Linear(4096, 1000))
        ]))
    
        # DECONV PART.
        self.deconv_pool5 = nn.MaxUnpool2d(kernel_size=3,
                                           stride=2,
                                           padding=0)
        self.deconv_act5 = nn.ReLU()
        self.deconv_conv5 = nn.ConvTranspose2d(256,
                                               384,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=False)
        
        self.deconv_act4 = nn.ReLU()
        self.deconv_conv4 = nn.ConvTranspose2d(384,
                                               384,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=False)
        
        self.deconv_act3 = nn.ReLU()
        self.deconv_conv3 = nn.ConvTranspose2d(384,
                                               256,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=False)
        
        self.deconv_pool2 = nn.MaxUnpool2d(kernel_size=3,
                                           stride=2,
                                           padding=1)
        self.deconv_act2 = nn.ReLU()
        self.deconv_conv2 = nn.ConvTranspose2d(256,
                                               96,
                                               kernel_size=5,
                                               stride=2,
                                               padding=0,
                                               bias=False)
        
        self.deconv_pool1 = nn.MaxUnpool2d(kernel_size=3,
                                           stride=2,
                                           padding=1)
        self.deconv_act1 = nn.ReLU()
        self.deconv_conv1 = nn.ConvTranspose2d(96,
                                               3,
                                               kernel_size=7,
                                               stride=2,
                                               padding=1,
                                               bias=False)
        
    def forward(self, x):
        
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                self.feature_outputs[i] = x
                self.switch_indices[i] = indices
            else:
                x = layer(x)
                self.feature_outputs[i] = x
            
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def forward_deconv(self, x, layer):
        if layer < 1 or layer > 5:
            raise Exception("ZFnet -> forward_deconv(): layer value should be between [1, 5]")
        
        x = self.deconv_pool5(x,
                              self.switch_indices[12],
                              output_size=self.feature_outputs[-2].shape[-2:])
        x = self.deconv_act5(x)
        x = self.deconv_conv5(x)
        
        if layer == 1:
            return x
        
        x = self.deconv_act4(x)
        x = self.deconv_conv4(x)
        
        if layer == 2:
            return x
        
        x = self.deconv_act3(x)
        x = self.deconv_conv3(x)
        
        if layer == 3:
            return x
        
        x = self.deconv_pool2(x,
                              self.switch_indices[5],
                              output_size=self.feature_outputs[4].shape[-2:])
        x = self.deconv_act2(x)
        x = self.deconv_conv2(x)
     
        if layer == 4:
            return x
        
        x = self.deconv_pool1(x,
                              self.switch_indices[2],
                              output_size=self.feature_outputs[1].shape[-2:])
        x = self.deconv_act1(x)
        x = self.deconv_conv1(x)
        
        if layer == 5:
            return x

model = ZFNet()
summary(model, (3, 224, 224))


#Download pretrained model.
urllib.request.urlretrieve('https://github.com/osmr/imgclsmob/releases/download/v0.0.395/zfnet-1727-d010ddca.pth.zip',
                           'zfnet-1727-d010ddca.pth.zip')

# Unzip 'zfnet-1727-d010ddca.pth.zip'.
with ZipFile('zfnet-1727-d010ddca.pth.zip', 'r') as zip_ref:
    zip_ref.extractall()
    
# Load pretrained model parameters into our model.
load_model('zfnet-1727-d010ddca.pth', model)

# Delete 'zfnet-1727-d010ddca.pth' and 'zfnet-1727-d010ddca.pth.zip'.
os.remove('zfnet-1727-d010ddca.pth')
os.remove('zfnet-1727-d010ddca.pth.zip')



fig2 = plt.figure(figsize=(30,10))
torch.save(model, 'save/to/path/model.pt')
model = torch.load('load/from/path/model.pt')



# Create a dataset.

class CustomDataset(Dataset):
    
    def __init__(self, transform = None):
        
        self.transform = transform
        self.imgs = []
        self.imgsToDisplay = []
        
        current_dir = os.getcwd()
        image_dir = os.path.join(current_dir, 'img/')
        img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    
        for img in img_files:
            # Read image.
            img = cv2.imread(img)
            
            self.imgsToDisplay.append(img)
            
            # Apply transformations.
            if self.transform is not None:
                img = self.transform(img)
                
            self.imgs.append(img)
            
    def __getitem__(self, index):    
        return self.imgs[index]
        
    def __len__(self):
        return len(self.imgs)


# Apply preprocessing step on dataset.
transformations = transforms.Compose([transforms.ToPILImage(), 
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                           std = [0.229, 0.224, 0.225])
                                     ])

custom_dataset = CustomDataset(transform=transformations)
test_loader = DataLoader(dataset=custom_dataset,
                         batch_size=1)





model.eval()
with torch.no_grad():
    
    for i, image in enumerate(test_loader):
        probs = torch.nn.Softmax(dim=-1)(model(image))
        
        probability, class_idx = torch.max(probs, 1)
        class_name = class_names[class_idx]
         
        fig2.add_subplot(1,4,i+1)
        plt.imshow(cv2.cvtColor(custom_dataset.imgsToDisplay[i], cv2.COLOR_BGR2RGB))
        plt.title("Class: " + class_name + ", probability: %.4f" % probability, fontsize=13)
        plt.axis('off')

        plt.text(0, 240, 'Top-5 Accuracy:')
        x, y = 10, 260
        
        for idx in np.argsort(probs.numpy())[0][-5::][::-1]:
            s = '- {}, probability: {:.4f}'.format(class_names[idx], probs[0, idx])
            plt.text(x, y, s=s, fontsize=10)
            y += 20
        print()


fig2 = plt.figure(figsize=(60,60))

model.eval()
count = 0
with torch.no_grad():
    for i, image in enumerate(test_loader):
        probs = torch.nn.Softmax(dim=-1)(model(image))
        for j in range(1,6):
            count += 1
            ax = fig2.add_subplot(4,5, count)
            ax.set_title("Layer {}".format(j), fontsize= 30)
            plt.axis('off')
            # Channel 3 of the image.
            plt.imshow(model.forward_deconv(model.feature_outputs[12], j).detach().numpy()[0, 2, :])
