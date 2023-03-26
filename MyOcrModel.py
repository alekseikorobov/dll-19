from typing import List
from MyDetection import MyDetection
from MyRecognizer import MyRecognizer
from utils.my_utils import sorting_bounding_box, convert_to_index_box,get_text_from_points,image_box_show_1,image_box_show, sorting_bounding_box_v2
from config_rec import all_alph
import cv2
from PIL import Image
import numpy as np


class MyOcrModel:
    def __init__(self,path_model_detect, model_path, all_alph, version_detect = 'v2') -> None:
        self.detect_model = MyDetection(path_model_detect, version_detect=version_detect)
        self.rec_model = MyRecognizer(model_path, all_alph)
        self.isRecording = False
        self.images, self.boxes = None, None
        
    def get_image_with_box_from_path(self, path_image):
        img = Image.open(path_image).convert("RGB")
        return self.get_image_with_box(img)
    
    def get_image_with_box(self, img:Image.Image):
        # img = Image.open(path_image).convert("RGB")
        if not self.isRecording:
            self.images, self.boxes = self.detect_model.detect_images(img)
        color = (255,0,0)
        for box in self.boxes:
            p1,p2 = (int(box[0]),int(box[1])),(int(box[2]),int(box[3]))
            img = np.asarray(img)            
            img = cv2.rectangle(img,p1,p2,color,1)
        return img

    
    def recognize_image(self, image:Image.Image)->List[str]:
        self.isRecording = False
        self.images, self.boxes = self.detect_model.detect_images(image)
        self.isRecording = True
        
        return self._rec_image_get_texts(self.images, self.boxes)

    def _rec_image_get_texts(self, images, boxes):
        text_boxes = []
        for img, boxe in zip(images,boxes):
            img = self.rec_model.processing_image(img)
            text = self.rec_model.get_text_from_image(img)
            text_boxes.append([text, boxe.numpy()])
        
        text_boxes = sorting_bounding_box_v2(text_boxes)
        result_texts = get_text_from_points(text_boxes)
        
        return result_texts
    
    def recognize_image_file(self, path_image:str)->List[str]:        

        images, boxes = self.detect_model.detect_images_path(path_image)
        
        return self._rec_image_get_texts(images, boxes)