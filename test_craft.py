from torch.utils.data import DataLoader
#from my_dataset import MyDataset
#from my_model import CRAFT
import torch.nn as nn
import torch
import os
from collections import OrderedDict
import numpy as np
from CRAFT.craft import CRAFT
from CRAFT import craft_utils

from MyDataset import MyDataset

import torchvision.transforms as transforms

import matplotlib.pyplot as plt


if __name__ == '__main__':    
    device = 'cpu'
    dataset_path = './data'
    pretrained_path = './pretrained/craft_mlt_25k.pth'
    model_path = './models'

    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    
    count_elements = 1


    img_size = 256
    train_tfms = transforms.Compose([transforms.Resize(img_size),
                                     transforms.ToTensor()])
    
    dataset = MyDataset(count_elements, train_tfms)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for k in range(2):
    #     print('___{k}___')
    #     for img,mask in loader:
    #         print(f'{img.shape=}, {mask.shape=}')

    net = CRAFT(pretrained=True).to(device)
    net.load_state_dict(torch.load(os.path.join(model_path,'499.pth')))

    net.eval()
   
    for img, gt in loader:
        with torch.no_grad():
            y, _ = net(img)

        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        text_threshold = 0.5
        link_threshold = 0.2
        low_text = 0.2
        poly = True
        print(f'{score_text=}, {score_link=}')
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        #print(f'{boxes=}, {polys=}')
        
        plt.imshow(img[0].permute(1,2,0).numpy())
        plt.show()
        plt.imshow(y[0][...,0].numpy())
        plt.show()
