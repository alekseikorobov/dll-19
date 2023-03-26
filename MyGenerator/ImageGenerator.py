
import PIL
import numpy as np
import cv2
import random as rnd
from typing import List, Tuple
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
import numpy as np
from MyGenerator.Bbox import Bbox
import itertools
from collections import MutableSequence
from MyGenerator.parse_text import Format, parse_text_format
from tqdm import tqdm


# Thai Unicode reference: https://jrgraphix.net/r/Unicode/0E00-0E7F
TH_TONE_MARKS = [
    "0xe47",
    "0xe48",
    "0xe49",
    "0xe4a",
    "0xe4b",
    "0xe4c",
    "0xe4d",
    "0xe4e",
]
TH_UNDER_VOWELS = ["0xe38", "0xe39", "\0xe3A"]
TH_UPPER_VOWELS = ["0xe31", "0xe34", "0xe35", "0xe36", "0xe37"]

from itertools import product
import os
class Params:
    def __init__(self, conf:dict) -> None:
        if conf is None:
            conf = {}        
        self.fonts = conf.get('fonts',[])
        self.texts:List[str] = conf.get('texts',['привет'])
        self.text_colors = conf.get('text_colors',['#000000'])
        self.size_images = conf.get('size_images',[(0,0)])
        self.position_texts = conf.get('position_texts',[(0,0)])
        self.font_sizes = conf.get('font_sizes',[14])
        self.background_colors = conf.get('background_colors',['#ffffff'])
        self.use_box = conf.get('use_box',[False])
        self.box_colors = conf.get('box_colors',['#555555'])
        self.use_lines = conf.get('use_lines',[''])
        self.lines_colors = conf.get('lines_colors',['#555555'])   
        self.is_crop = conf.get('is_crop',[False])   
        self.is_simple_text = conf.get('is_simple_text',[False])
        self.is_scale = conf.get('is_scale',[False])
        self.scale_size = conf.get('scale_size',[(0,0)])
    
    
    @classmethod
    def get_params(*kwargs):
        p = Params(None)
        pars = kwargs[1]
        for i,k in enumerate(p.__dict__.keys()):            
            p.__dict__[k] = pars[i]            
        return p

class ImageGenerator:
    
    def __init__(self, confs:List[dict]) -> None:
        
        self.all_param_list:List[Params] = []
        for conf in confs:
            
            iterables = []
            
            p = Params(conf)
            for k in p.__dict__.keys():
                iterables.append(list(set(p.__dict__[k] )))
                        #
            self.all_data = filter(self.strict_combination,
                                map(Params.get_params,
                                    product(*iterables)
                                )
                            )
            self.all_param_list.extend(list(self.all_data))
        #print(f'self.all_param_list {len(self.all_param_list)}')
        #print('get_word_formated_list...')
        self.word_formated_list = self.get_word_formated_list()
        
        #print('get_max_pix_size...')
        self.max_pix_size_width, self.max_pix_size_height = self.get_max_pix_size()
        
        #print('get_max_pix_size_from_param...')
        self.max_pix_size_param_width, self.max_pix_size_param_height = self.get_max_pix_size_from_param()
        ##print(f'{self.max_pix_size_param_width, self.max_pix_size_param_height=}')
        
        # assert self.max_pix_size_param_width>=self.max_pix_size_width, f'''заданная ширина картинки должа быть больше либо равна фактическому размеру картинки для слова. 
        # Максимальный заданный размер {self.max_pix_size_param_width=}, фактический размер - {self.max_pix_size_width=}, необходимо поменять параметр size_images'''
        
        # assert self.max_pix_size_param_height>=self.max_pix_size_height, f'''заданная высота картинки должа быть больше либо равна фактическому размеру картинки для слова. 
        # Максимальный заданный размер {self.max_pix_size_param_height=}, фактический размер - {self.max_pix_size_height=}, необходимо поменять параметр size_images'''
        
        
        ##print(f'{self.max_pix_size_width=}')
        ##print(f'{self.max_pix_size_height=}')
        
        self.size = len(self.all_param_list)
        self.size_word = len(self.word_formated_list)

    def get_max_pix_size(self):
        max_width = max(map(lambda x: x[1].pix_size_width,self.word_formated_list))
        max_height = max(map(lambda x: x[1].pix_size_height,self.word_formated_list))
        return max_width, max_height
    
    def get_max_pix_size_from_param(self):
        max_width = max(map(lambda x: x.size_images[0],self.all_param_list))
        max_height = max(map(lambda x: x.size_images[1],self.all_param_list))
        return max_width, max_height

    def get_word_formated_list(self)->List[Tuple[str, Format, Params]]: #result (word, format, params)
        res = []
        for params in tqdm(self.all_param_list):
            # if idx % 1000 == 0:
            ##     print(f'{idx=}')
                
            if params.is_simple_text:
                word = params.texts.strip()
                if word == '': continue
                format = Format()
                image_font = self.get_image_font(params, format)
                bbox = image_font.getbbox(word)
                format.pix_size_height = bbox[3]
                format.pix_size_width = bbox[2]
                # format.pix_size_width = sum(
                #     [self._compute_character_width(image_font, p)
                #     for p in word.strip() ]
                # )
                res.append((word,format,params))
            else:                
                text_lines:List[str] = params.texts.split("\n")
                for text_line in text_lines:
                    if text_line.strip() == '': continue
                    words = parse_text_format(text_line)
                    for word, format in words:
                        word = word.strip()
                        if word == '': continue                
                        image_font = self.get_image_font(params, format)
                        format.pix_size_width = sum(
                            [self._compute_character_width(image_font, p)
                            for p in word.strip() ]
                        )
                        bbox = image_font.getbbox(word)
                        format.pix_size_height = bbox[3]

                        res.append((word,format,params))
        return res
    
    def strict_combination(self, data: Params):
        '''
            метод для пропуска значений, сочетание которые недопустимо
            например, когда цвет текста и фона совпадают
        '''
        # условия по которым отсекаются сочетания
        return not \
            ( data.text_colors == data.background_colors \
                # or (len(text.split('\n')) > 2 
                #     and position_text[1]>=120  
                #     and size_image[1]<=120)
                
                # or (len(text.split('\n')) > 8
                #     and size_image[1]<=280)
                
                # or (max([len(x) for x in  text.split('\n')]) > 75
                #     and size_image[0]<=800
                #     and font_size > 17
                #     #and position_text[1]>40
                #     )   
                
                             
            )
             
    
    def _compute_character_width(self,image_font: ImageFont, character: str) -> int:
        if len(character) == 1 and (
            "{0:#x}".format(ord(character))
            in TH_TONE_MARKS + TH_UNDER_VOWELS + TH_UNDER_VOWELS + TH_UPPER_VOWELS
        ):
            return 0
        # Casting as int to preserve the old behavior
        return round(image_font.getlength(character))

    def generate_horizontal_text(self, params: Params) -> Tuple[Image.Image, List[Bbox], List[str]]:
        space_width_default=1.5
        stroke_width = 0
        stroke_fill = "#000000"        
        texts:List[str] = [] #result list words        
        global_image_font = ImageFont.truetype(font=params.fonts, size=params.font_sizes)        
        space_width = int(round(global_image_font.getlength(" ")) * space_width_default)
        tabl_width = int(round(global_image_font.getlength("\t") * 3))        
        text_lines:List[str] = params.texts.split("\n")        
        boxs: List[Bbox] = [] #result boxses        
        txt_img = Image.new("RGB", params.size_images, params.background_colors)
        txt_img_draw = ImageDraw.Draw(txt_img)
        
        
        all_text_height = 0
        max_text_height = 0
        for line_index,text_line in enumerate(text_lines):
            words = parse_text_format(text_line)
            new_line = True
            text_line = ''.join(map(lambda c: (' ' if c[1] is not None and c[1].is_space else '') + c[0], words))

            #for word, format in words:
            
            for word in words:
                ft = word[1] 
                if ft is None:
                    ft = Format()
                ft.pix_size_width = sum(
                    [self._compute_character_width(global_image_font, p)
                     for p in word[0].strip() ]
                )
                word[1] = ft
            
            ##print(f'{words=}')
            
            text_line = text_line.strip(' ')
            if text_line == '': continue
            ##print(f'{text_line=}')
            
            
            bbox = global_image_font.getbbox(text_line)
            text_height = bbox[3] - bbox[1]
            max_text_height = max(max_text_height, text_height)
            
            interval = line_index * 5 #межстрочный интервал
            
            all_text_height += max_text_height

            ##print(f'{words=}')            
            count_w = 0
            x_pos = params.position_texts[0]
            image_font = None
            for i, (word, format) in enumerate(words):
                
                if format.size is not None:
                    image_font = ImageFont.truetype(font=params.fonts, size=int(format.size))        
                else:
                    image_font = global_image_font
                    
                format.pix_size_width = sum(
                    [self._compute_character_width(image_font, p)
                     for p in word.strip() ]
                )
                
                space_width = int(round(image_font.getlength(" ")) * space_width_default)
                bbox = image_font.getbbox(text_line)
                text_height = bbox[3] - bbox[1]
                if text_height > max_text_height:
                    all_text_height -= max_text_height
                    max_text_height = text_height
                    all_text_height += max_text_height
                
                
                count_tabl = len(list(itertools.takewhile(lambda x: x == '\t',word)))
                x_pos += count_tabl * tabl_width
                if format.is_space:
                    x_pos += space_width

                #####print(f'{word=}')
                #it = len(word)
                word = word.strip()
                # if word != '':
                #     texts.append(word)
                    
                count_w += int(1 if format is not None and format.is_space else 0)                
                
                y_pos = params.position_texts[1]  + (max_text_height * line_index) + interval
                ##print(f'{word=} {now_position=} {x_pos1, y_pos =}')
                txt_img_draw.text(
                    (x_pos, y_pos),
                    word,
                    fill= format.color if format is not None and format.color is not None else params.text_colors,
                    font=image_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )                
                ##print(f'{new_line=}')
                if word != '':
                    x_max = x_pos + 1 + format.pix_size_width
                    if not format.is_space and len(boxs)>0 and not new_line:
                        box = boxs[len(boxs)-1]
                        box.x_max = x_max
                        texts[len(texts)-1] += word
                    else:
                        box = Bbox(x_min = x_pos-1
                                ,y_min = params.position_texts[1] + bbox[1]-1 + (max_text_height * line_index) + interval
                                ,x_max = x_max
                                ,y_max = params.position_texts[1] + bbox[3]+1 + (max_text_height * (line_index)) + interval
                                )
                        boxs.append(box)
                        texts.append(word)
                    new_line = False                
                x_pos += format.pix_size_width
                
            if params.use_box:
                p1,p2 = box.get_p1_p2()
                #txt_img_draw.rectangle((p1,p2),fill='#ff0000')
                txt_img_draw.rounded_rectangle((p1,p2), outline = params.box_colors)
                
        if params.use_lines != '' and len(params.use_lines) == 4:
            x_min = min([box.x_min for box in boxs])-5
            y_min = min([box.y_min for box in boxs])-5
            y_max = max([box.y_max for box in boxs])+5
            x_max = max([box.x_max for box in boxs])+5
            
            w_top, w_right, w_bottom, w_left =  [int(x) for x in params.use_lines]
            
            if w_left > 0:
                txt_img_draw.line(((x_min,y_min),(x_min,y_max)),fill=params.lines_colors,width=w_left)
                
            if w_bottom > 0:
                txt_img_draw.line(((x_min,y_max),(x_max,y_max)),fill=params.lines_colors,width=w_bottom)
                
            if w_right > 0:
                txt_img_draw.line(((x_max,y_min),(x_max,y_max)),fill=params.lines_colors,width=w_right)
            
            if w_top > 0:
                txt_img_draw.line(((x_min,y_min),(x_max,y_min)),fill=params.lines_colors,width=w_top)
        
        assert len(boxs) == len(texts), f'count box {len(boxs)}, count texts {len(texts)} must be equeal'
                
        return txt_img, boxs, texts

    def get_image_font(self, params: Params, format:Format):
        image_font = None
        if format.size is not None:
            image_font = ImageFont.truetype(font=params.fonts, size=int(format.size))        
        else:
            image_font = ImageFont.truetype(font=params.fonts, size=params.font_sizes) 
        
        return image_font


    def generate_horizontal_text_crop(self, params: Params) -> Tuple[List[Image.Image], List[Bbox], List[str]]:       
        '''
            return:
            - list image [(h, w, c)]
            - list Bbox None
            - list text ['str']
            
        '''
        texts:List[str] = [] #result list words        
        text_lines:List[str] = params.texts.split("\n")
        images = []
        
        for text_line in text_lines:
            
            if text_line.strip() == '': continue

            words = parse_text_format(text_line)

            for word, format in words:
                word = word.strip()
                if word == '': continue
                
                image_font = self.get_image_font(params, format)
                
                left, top, right, bottom = image_font.getbbox(word)
                #print(f'{left, top, right, bottom=}')
                txt_img = None
                if params.is_scale:
                    txt_img = Image.new("RGB", (right, bottom-top) , params.background_colors)                
                else:
                    txt_img = Image.new("RGB", (self.max_pix_size_param_width, self.max_pix_size_param_height) , params.background_colors)
                

                txt_img_draw = ImageDraw.Draw(txt_img)
                                
                txt_img_draw.text(
                    (0, -top),
                    word,
                    fill = format.color if format is not None and format.color is not None else params.text_colors,
                    font = image_font
                )
                if params.is_scale:
                    # w = params.scale_size[0]
                    # h = params.scale_size[1]
                    # if len(word) <= 3:
                    #     w = 32
                    # elif len(word) <= 7:
                    #     w = 64
                    # elif len(word) <= 14:
                    #     w = 128
                    # txt_img = txt_img.resize((w,h))
                    txt_img = self.image_resize(txt_img,params.scale_size[0],params.scale_size[1] )
                    #txt_img = cv2.resize(np.asarray(txt_img),params.size_images)
                    
                texts.append(word)
                images.append(txt_img) #.crop(txt_img.getbbox())

        assert len(images) == len(texts), f'count images {len(images)}, count texts {len(texts)} must be equeal'
                
        return images, None, texts
    
    def image_resize(self, image:Image.Image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        #print(f'{image.size=}')
        (w, h) = image.size# shape[:2]

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
        #print(f'{dim=}')

        # resize the image
        resized = image.resize(dim) #, interpolation = inter

        # return the resized image
        return resized

    def get_by_word_index(self, index) -> Tuple[Image.Image, str]:
        
        word,format,params = self.word_formated_list[index]        
                
        image_font = self.get_image_font(params, format)
        
        left, top, right, bottom = image_font.getbbox(word)
        #print(f'{left, top, right, bottom=}')
        
        txt_img = None
        if params.is_scale:
            txt_img = Image.new("RGB", (right, bottom-top) , params.background_colors)                
        else:
            txt_img = Image.new("RGB", (self.max_pix_size_param_width, self.max_pix_size_param_height) , params.background_colors)
        
        #txt_img = Image.new("RGB", (right, bottom) , params.background_colors)

        txt_img_draw = ImageDraw.Draw(txt_img)
                        
        txt_img_draw.text(
            (0, -top),
            word,
            fill = format.color if format is not None and format.color is not None else params.text_colors,
            font = image_font
        )
        if params.is_scale:
            # h = params.scale_size[1]
            # w = params.scale_size[0]
            # if len(word) <= 3:
            #     w = 32
            # elif len(word) <= 7:
            #     w = 64
            # elif len(word) <= 14:
            #     w = 128

            # txt_img = txt_img.resize((w,h))
            txt_img = self.image_resize(txt_img,params.scale_size[0],params.scale_size[1] )
            #txt_img = cv2.resize(np.asarray(txt_img),params.size_images)
        return txt_img, word
        

    def get_by_index(self, index):        
        params:Params = self.all_param_list[index]
        if not params.is_crop:
            img, boxs, texts = self.generate_horizontal_text(params)
        else:
            #print(f'generate_horizontal_text_crop')
            img, boxs, texts = self.generate_horizontal_text_crop(params) 
        return img, boxs, texts
