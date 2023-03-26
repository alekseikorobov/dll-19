import MyDatasetRec
import importlib
importlib.reload(MyDatasetRec)
from MyDatasetRec import MyDatasetRec
from torch.utils.data import DataLoader
from config_rec import all_alph
import MyModelRec as my_model
import torch
from itertools import groupby
from MyGenerator.TorchTextDict import TorchTextDict
import numpy as np
import cv2
from PIL import Image
from MyDatasetRec import MyDatasetRec

class MyRecognizer:
    def __init__(self, path_model:str, all_alph:str, word_size = 100) -> None:
        #self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')
        num_classes = len(all_alph)+1
        self.model = my_model.CRNN_v1(imgH=32,in_channels=3, nclass=num_classes, gru_size=256)
        path = path_model
        self.model = my_model.load_model(self.model, path, self.device)
        self.model.eval()
        self.model.to(self.device)
        self.torch_text_dict = TorchTextDict(all_alph, word_size = word_size)
        
        self.transforms = MyDatasetRec.get_transform(self)
        
        #TODO: add resize image to (160 ,32)
        
        
        #self.torch_text_dict = torch_text_dict
    def image_resize(self, image:np.ndarray, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized
    
    def get_text_from_image(self, img: torch.Tensor):
        with torch.no_grad():
            x_train = img
            #print(f'{x_train.shape=}')
            x_train = x_train.to(self.device)
            y_pred = self.model(x_train)
            #print(f'{y_pred.shape}')            
            blank_label = 0
            _, max_index = torch.max(y_pred, dim=2)
            
            raw_prediction = list(max_index[:, 0].detach().cpu().numpy())
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
            txt_predict = self.torch_text_dict.get_label(prediction)
            return txt_predict
        
    def processing_image(self,img_np:np.ndarray):
        img1 = self.image_resize(img_np, width = None, height = 32)

        #print(f'{img1.shape=}')

        img2 = Image.fromarray(img1)

        img_tensor,_=self.transforms(img2,None)
        
        return img_tensor.unsqueeze(0)