import os
from typing import List, Tuple 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor,MaskRCNN
import torch
import os, errno
import common.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import MyModel as my_model

class MyDetection:
    to_image = torchvision.transforms.ToPILImage()
    def __init__(self, path_model:str, device = None,version_detect='v1')-> None:
        self.device = device if device is not None else 'cpu'
        num_classes = 2
        if not os.path.exists(path_model):
            raise ValueError(f'Not exists file path {path_model}')

        if version_detect == 'v1':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            self.model.roi_heads.mask_predictor  = None
            state_dict = torch.load(path_model, map_location=torch.device(self.device))
            self.model.load_state_dict(state_dict['model'])
        else:
            self.model = my_model.load_model(path_model, num_classes, self.device)
        self.model.to(self.device)
        self.model.eval()
        self.transform = self.get_transform()
    
    # def detect_iamge(self,img):
    #     img = to_image(x[0])
    #     x = x[0].to(device)
    #     predict = model([x])
    
    
    def get_transform(self):
        transforms = []
        #transforms.Lambda(lambda image: image.convert("RGB")),
        #transforms.ConvertImageDtype(torch.float),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # converts the image, a PIL image, into a PyTorch Tensor
        #transforms.append(T.torchvision.transforms.PILToTensor())
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))        
        return T.Compose(transforms)
    
    def detect_image(self,img:Image.Image):
        with torch.no_grad():            
            img_tensor,_ = self.transform(img,None)
            img_tensor = img_tensor.to(self.device)
            predict = self.model([img_tensor])
            return predict[0]
    
    def detect_images_path(self,path_image):
        img = Image.open(path_image).convert("RGB")
        return self.detect_images(img)
    
    def detect_images(self,img:Image.Image) -> Tuple[List[np.ndarray],torch.Tensor]:
        '''
            return List[np array] 
        '''
        result_images = []
        res = self.detect_image(img)
        img1 = np.array(img)
        #plt.imshow(img1[10:30,10:30])
        #img1 = np.array(img)
        for box in res['boxes']:
            #box = y[0]['boxes'].view(-1)
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            #print(box)
            
            img_part = img1[y1:y2,x1:x2]
            result_images.append(img_part)
            # plt.imshow(img_part)
            # plt.show()
        return result_images,res['boxes']