from torch.utils.data import DataLoader
#from my_dataset import MyDataset
#from my_model import CRAFT
import torch.nn as nn
import torch
import os
from collections import OrderedDict
import numpy as np
from CRAFT.craft import CRAFT

from MyDataset import MyDataset

import torchvision.transforms as transforms



def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

if __name__ == '__main__':    
    device = 'cpu'
    dataset_path = './data'
    pretrained_path = './pretrained/craft_mlt_25k.pth'
    model_path = './models'

    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    
    count_elements = 2


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
    #net.load_state_dict(torch.load(os.path.join(model_path,'173.pth')))
    # net.load_state_dict(copyStateDict(torch.load(pretrained_path, map_location=device)))
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer=torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),1e-7,
                               momentum=0.95,
                               weight_decay=0)
    # if not os.path.exists(model_path):
    #     os.mkdir(model_path)
    
    for epoch in range(1,500):
        epoch_loss = 0
        for img, gt in loader:
            #img = data['img'].to(device)
            #gt = data['gt'].to(device)
            img = img#.permute(0,2,3,2)
            #print(f'{img.shape=}')
            #break;
            # forward
            y, _ = net(img)
            #print(f'{y.shape=} {gt.shape=}')
#            break
            loss = criterion(y, gt)
            #break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
        print('epoch loss_'+str(epoch),':',epoch_loss/len(loader))
        torch.save(net.state_dict(), os.path.join(model_path,str(epoch)+'.pth'))
        if os.path.isfile(os.path.join(model_path,str(epoch-1)+'.pth')):
            os.remove(os.path.join(model_path,str(epoch-1)+'.pth'))

