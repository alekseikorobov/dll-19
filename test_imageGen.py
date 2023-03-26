
#%%
import MyGenerator.ImageGenerator as im
import matplotlib.pyplot as plt
import importlib
importlib.reload(plt)
importlib.reload(im)
from MyGenerator.ImageGenerator import ImageGenerator
import matplotlib.pyplot as plt

config = [{
    'fonts':['dataset/MyGenerator/font.ttf']
    , 'texts':[f'''
    <format color=#0000ff>class</format> MyClass
    {{
    \tpublic string P{{get;set;}}
    }}
    ''']
    , 'text_colors':['#0000ff']
    , 'size_images':[(200,100)]
    , 'position_texts':[(5,2)]
    , 'font_sizes':[12]
    , 'background_colors':['#ffffff']
    , 'use_box':[False]
    , 'box_colors':['#555555']
    #, 'use_lines':['','|','-','\\','/']
    , 'use_lines':['0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']
}]


i = ImageGenerator(config)

img,box,text = i.get_by_index(0)

#plt.imshow(img)
plt.plot([1])

plt.show()

# %%
