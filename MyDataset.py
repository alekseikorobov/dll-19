

import numpy as np
import torch
from torch.utils.data import Dataset
import common.transforms as T
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from MyGenerator.ImageGenerator import ImageGenerator


class MyDataset(Dataset):
    def __init__(self,conf,is_train = False) -> None:
        self.generator = ImageGenerator(conf)
        self.transforms = self.get_transform(is_train)
        
    def get_transform(self, train):
        transforms = []
        #transforms.Lambda(lambda image: image.convert("RGB")),
        #transforms.ConvertImageDtype(torch.float),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # converts the image, a PIL image, into a PyTorch Tensor
        #transforms.append(T.torchvision.transforms.PILToTensor())
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        #if train:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            #transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def __getitem__(self, index):
        
        img, boxs, text = self.generator.get_by_index(index)
        
        boxes = [box.array() for box in boxs]        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        masks = [box.get_h_w(img.size) for box in boxs]
        
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        #print(f'{masks.shape=}')
        
        num_objs = len(boxs)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks #(UInt8Tensor[N, H, W])
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return self.generator.size
    


if __name__ == '__main__':
    import common.transforms as T
    
    def get_transform(train):
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)    
    train_tfms = get_transform(False)
    
    conf = {
        'fonts':['TextRecognitionDataGenerator/tests/font.ttf']
        , 'texts':['Привет Мир!!!!\nПока мир!\nввв','dfdsfsd']
        , 'text_colors':['#00ffff','#ff0000']
        , 'size_images':[(280,280)]
        , 'position_texts':[(20,40),(2,4),(40,50)]
        , 'font_sizes':[32]
        , 'background_colors':['#000000','#ffffff']
    }
    
    dataset = MyDataset(conf,train_tfms)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for im,l in loader:
        print(f'{im.shape},{l}')
        break
