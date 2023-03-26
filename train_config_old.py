text1 = '''Почему он используется?
Давно выяснено, что при оценке дизайна и композиции читаемый текст
мешает сосредоточиться. Lorem Ipsum используют потому, что тот
обеспечивает более или менее стандартное заполнение шаблона,
а также реальное распределение букв и пробелов в абзацах, которое
не получается при простой дубликации "Здесь ваш текст.. Здесь ваш текст..
Здесь ваш текст.." Многие программы электронной вёрстки и редакторы
HTML используют Lorem Ipsum в качестве текста по умолчанию, так
что поиск по ключевым словам "lorem ipsum" сразу показывает, как много
веб-страниц всё ещё дожидаются своего настоящего рождения. За прошедшие годы
текст Lorem Ipsum получил много версий. Некоторые версии появились по ошибке,
некоторые - намеренно (например, юмористические варианты).'''

text2 = '''Почему он используется?
Давно выяснено, что при оценке дизайна и композиции читаемый текст
мешает сосредоточиться. Lorem Ipsum используют потому, что тот







веб-страниц всё ещё дожидаются своего настоящего рождения. За прошедшие годы
текст Lorem Ipsum получил много версий. Некоторые версии появились по ошибке,
некоторые - намеренно (например, юмористические варианты).'''

configuration = [
    {
        'fonts':['dataset/MyGenerator/font.ttf']
        , 'texts':[text1,text2]
        , 'text_colors':['#0000ff','#00ff00','#ff0000','#ff00ff','#000000','#ffffff'] #,'#ff0000'
        , 'size_images':[(800,800)]
        , 'position_texts':[(20,40),(2,4),(2,100),(2,300),(2,500)] #,
        , 'font_sizes':[17,18,22]
        , 'background_colors':['#ffffff','#000000']
    },
    {
        'fonts':['dataset/MyGenerator/font.ttf']
        , 'texts':[
                'Привет мир!\nПока Пока Пока\nПока Пока Пока',               
                ]
                    
        , 'text_colors':['#0000ff','#00ff00','#ff0000','#ff00ff','#000000','#ffffff'] #,'#ff0000'
        , 'size_images':[(280,280)]
        , 'position_texts':[(2,4),(20,40),(40,40),(100,100)] #,
        , 'font_sizes':[15,17,18,30]
        , 'background_colors':['#ffffff','#000000']
        , 'use_box':[True, False]
        , 'box_colors':['#555555']
    },
    {
        'fonts':['dataset/MyGenerator/font.ttf']
        , 'texts':['select *\nFROM [table]'
                ,"#0000ff','#00ff00'\n'#ff0000','#ff00ff'"
                ]
                    
        , 'text_colors':['#0000ff','#00ff00','#ff0000','#ff00ff','#000000','#ffffff'] #,'#ff0000'
        , 'size_images':[(800,800)]
        , 'position_texts':[(2,4),(20,40),(40,40),(100,100)] #,
        , 'font_sizes':[15,17,18,30]
        , 'background_colors':['#ffffff','#000000']
        
        # , 'use_sintaxis_color':[True,False]
        # , 'use_noze':[0, 20, 40] #value in percent
        , 'use_lines':['0004','0002','']
        , 'use_box':[False]
        , 'box_colors':['#555555']
        # , 'use_special_chars':[True, False]
    },
    {
        'fonts':['dataset/MyGenerator/font.ttf']
        , 'texts':['select *\nFROM [table]\nwhere userID = 123 and [t].fix = "121"'
                ,'select *\nFROM [table]\nwhere userID = 123 and [t].fix = "121"\nGROUP BY [u].USERID\nORDER BY Count(*)'
                ]
                    
        , 'text_colors':['#0000ff','#00ff00','#ff0000','#ff00ff','#000000','#ffffff'] #,'#ff0000'
        , 'size_images':[(800,800)]
        , 'position_texts':[(2,4),(20,40),(40,40),(10,100)] #,
        , 'font_sizes':[15,17,18]
        , 'background_colors':['#ffffff','#000000']
        
        # , 'use_sintaxis_color':[True,False]
        # , 'use_noze':[0, 20, 40] #value in percent
        , 'use_lines':['0000','2000','0200','0020','0002','2300','0230','0023','0032']
        , 'use_box':[False]
        , 'box_colors':['#555555']
        # , 'use_special_chars':[True, False]
    },
    {
    'fonts':['dataset/MyGenerator/font.ttf']
    , 'texts':['''import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import MyGenerator.utils.transforms as T
from util.augmentation import BaseTransform
from dataset.MyDataset import MyDataset'''               
                ]
                    
        , 'text_colors':['#0000ff','#00ff00','#ff0000','#ff00ff','#000000','#ffffff'] #,'#ff0000'
        , 'size_images':[(800,800)]
        , 'position_texts':[(2,4),(20,40),(40,40)] #,
        , 'font_sizes':[15,17,18]
        , 'background_colors':['#ffffff','#000000']
        
        # , 'use_sintaxis_color':[True,False]
        # , 'use_noze':[0, 20, 40] #value in percent
        , 'use_lines':['0000','2000','0200','0020','0002','2300','0230','0023','0032']
        , 'use_box':[False]
        , 'box_colors':['#555555']
        # , 'use_special_chars':[True, False]
    }
]