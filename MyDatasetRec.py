

from typing import Dict, Optional, Tuple
import numpy as np
import torch
from config_rec import all_alph
from torch.utils.data import Dataset
import common.transforms as T
import torchvision.transforms as transforms
from torch import nn, Tensor
from torch.utils.data import DataLoader
from MyGenerator.ImageGenerator import ImageGenerator
from MyGenerator.TorchTextDict import TorchTextDict
import operator

class MySameSizeTransform(nn.Module):
    def __init__(self, fix_size) -> None:
        super().__init__()
        self.fix_size = fix_size
    
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        #print(image.dtype)

        c_fact = image.shape[0]
        h_fact = image.shape[1]
        w_fact = image.shape[2]
        w_new = self.fix_size[1] - w_fact
        mask = torch.ones((c_fact, h_fact, w_new)) * -1
        
        #mask = mask.type(image.dtype)
        res = torch.cat((image,mask),2)
        #print(res.dtype)  
        return res, target
    
class MyTruncateTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, image: Tensor) -> Tensor:
        
        assert image.dim() == 3, 'shape must be c, h, w'
        
        left_part_index = operator.indexOf(image[0][0],-1)

        res = image[:,:,:left_part_index]
        # if res.dtype != torch.uint8:
        #     res = res.type(torch.uint8)
        return res

class MyDatasetRec(Dataset):
    '''
    dataset for recognition
    '''
    def __init__(self,conf, all_alph, fix_size = (32, 500), use_my_transform = True) -> None:
        self.generator = ImageGenerator(conf)
        #self.use_my_transform = use_my_transform
        self.my_same_size_transform = MySameSizeTransform(fix_size=fix_size)
        self.transforms = self.get_transform(self)
        #self.counts_word = 200
        self.torch_text_dict = TorchTextDict(all_alph, word_size = 100)
        
    @staticmethod
    def get_transform(self,fix_size = (32, 500)):
        transforms = []
        #transforms.Lambda(lambda image: image.convert("RGB")),
        #transforms.ConvertImageDtype(torch.float),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # converts the image, a PIL image, into a PyTorch Tensor
        #transforms.append(T.torchvision.transforms.PILToTensor())
        #transforms.append(T.res ())
        transforms.append(T.PILToTensor())        
        transforms.append(MySameSizeTransform(fix_size=fix_size))
        #if train:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            #transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def __getitem__(self, index):
        
        img, text = self.generator.get_by_word_index(index)
        #print(f'{texts=}')       
        text_tensor = self.torch_text_dict.fit_transform_word(text)
        
        #texts_tensor = torch.from_numpy(texts_numpy).int()
        #texts_tensor - size CW, WS: CW - count words, WS - word size (count chars)

        #imgs - size B,C,H,W: B - batch (count images) must be equal - texts_tensor[CW],C- count chanal, H - hight, W - width        

        if self.transforms is not None:
            img, _ = self.transforms(img, None)

        return img, text_tensor

    def __len__(self):
        return self.generator.size_word
    


if __name__ == '__main__':
    conf = {
        'fonts':["example/TextBPNPlusPlus/dataset/MyGenerator/font.ttf"],
        'is_crop':[True],
        'texts':
            '''привет всем, вот и я'''.split(' ')
        
        , 'is_simple_text':[True]
        , 'font_sizes':[8]
        , 'scale_size':[(None, 32)]
        , 'is_scale':[True]
    }
    my_trans = MyTruncateTransform()
    dataset = MyDatasetRec([conf],all_alph)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for im, l in loader:
        print(f'{im.shape=}')
        for im1,l1 in zip(im, l):
            print(f'before {im1.shape=}')
            im1 = my_trans(im1)
            print(f'after {im1.shape=}')
            print(f'after {im1=}')
        break
