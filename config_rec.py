import numpy as np
import json
import numpy as np

max_length = 8
count_sample = 200


fonts = ['example/TextBPNPlusPlus/dataset/MyGenerator/font.ttf'
        ,'example/TextBPNPlusPlus/dataset/MyGenerator/XO_Oriel_Bi.ttf'
        ,'example/TextBPNPlusPlus/dataset/MyGenerator/consolas/consola.ttf'
         ]

# dig_str = '1234567890'
# spec_str = '-=~@#$^&!№%*()_+[];\'/{}|:"<>?\\.,'
# eng_alph_str = 'qwertyuiopasdfghjklzxcvbnm'
# ru_alph_str =  'ёйцкнгшщзъфывплджэячмитьбю'
# all_alph = dig_str+spec_str+eng_alph_str+ru_alph_str
#all_alph = dig_str + eng_alph_str

dig_str = '1234567890'
spec_str = '-=~@#$^&!№%*()_+[];\'/{}|:"<>?\\.,'
eng_alph_str = 'qwertyuiopasdfghjklzxcvbnm'
ru_alph_str = 'ёйцукенгшщзхъфывапролджэячсмитьбю'
all_alph = dig_str+spec_str+eng_alph_str+ru_alph_str
print(f'{len(all_alph)=}')


json_file = 'utils/dict.json'
array = []
# with open(json_file, 'r' ) as f:
#     di = json.load(f)
#     for i in di:
#         if len(i) > 2 and len(i) < max_length:
#             text = ''
#             for c in i:
#                 if c in all_alph:
#                     text += c
#             if text != '':
#                 array.append(text)
#     print(f'{array[0:3]}')
#     #print(len(array))



def get_all_elements(all_alph, max_length, count_sample):
    res = []
    for i in range(max_length,0,-1):
        res.extend(
            ''.join(set(np.random.choice(np.array([c for c in all_alph]), i)))
            for _ in range(count_sample) 
        )

    #print(list(set(res)))
    return list(set(res))

#texts = get_all_elements(all_alph, max_length, count_sample)
#print(f'{len(texts)=}')

import os
def get_words(path,size=(2,8),count=100):
    res = set()
    if os.path.exists(path):
        with open(path,'r') as f:
            for line in f.readlines():
                text = line.strip().strip('\t').replace('\n','').lower()            
                if len(text)< size[0]:
                    continue
                for word in text.split():
                    if len(word)>=size[0] and len(word)<=size[1]:
                        text1 = ''
                        for c in word:
                            if c in all_alph:
                                text1 += c
                        if text1 != '':
                            res.add(text1)
                        
                    if len(res)>=count:
                        return list(res)
    else:
        print(f'WARN: path not exists - {path}')
    return list(res)
            
words = get_words('example/TextBPNPlusPlus/data/onegin.txt')

texts = ['1','3']
texts.extend(words)
texts.extend(array)
             #[0:count_sample])


texts = list(filter(lambda x: x != '', set(texts)))

# alph_fact = ''.join(sorted(set(''.join(texts))))
# all_alph = ''.join(sorted(all_alph))

# print(f'{alph_fact=}')
# print(f'{all_alph=}')


print(f'all - {len(texts)=}')


train_conf = [{
    'fonts':fonts,
    'is_crop':[True]
    , 'text_colors':[
                    #'#0000ff',
                     '#000000'
                     ,'#ffffff'
                     #, '#ff0000','#D8D9D9','#030303','#333333','#ffffaa','#2C6DBF','#4CC5AC','#9008CE','#2d01fd','#2b9ec9','#b62a15','#038003','#0505FE','#AFA4A4','#FD3D15','#FF00FF','#A59CA0'
                     ]
    , 'background_colors':[
                           '#000000',
                           '#ffffff'
                           #,'#1E1E1E','#1a1a1a','#ffffaa','#333333'
                           ]
    , 'texts':
        #texts
        np.random.choice(texts,300)
        
        # array[0:350]
        #list(map(str,np.random.random_integers(100,100000,100)))
        #['12345678901234567890123456789012345678901234567890'] #14 epoch
        #['deletedresponsiblepersonid'] #477 epoch
        #['isnull(cast(deletedresponsiblepersonid'] #407 epoch
    ,
    #'size_images':[((int(153/16)+1)*16 +2 ,18)] #for cnn_output_width = 19
    'size_images':[(max_length * 20 ,32)] #cnn_output_width * 16 + 2
    , 'is_simple_text':[True]
    , 'font_sizes':[12,14,16,18,22,23,24]
    , 'scale_size':[(None, 32)]
    , 'is_scale':[True]
},
#C# black theme
{
    'fonts':fonts,
    'is_crop':[True]
    , 'text_colors':['#2C6DBF','#4CC5AC','#9008CE','#D8D9D9']
    , 'background_colors':['#1a1a1a']
    , 'texts':

        np.random.choice(texts,300)
        
        #np.array(['12','13','14']).tolist()
        
    ,
    #'size_images':[((int(153/16)+1)*16 +2 ,18)] #for cnn_output_width = 19
    'size_images':[(max_length * 20 ,32)] #cnn_output_width * 16 + 2
    , 'is_simple_text':[True]
    , 'font_sizes':[12,14,16,18,22,23,24]
    , 'scale_size':[(None, 32)]
    , 'is_scale':[True]
},
#C# black white
{
    'fonts':fonts,
    'is_crop':[True]
    , 'text_colors':['#2d01fd','#2b9ec9','#b62a15','#038003','#030303']
    , 'background_colors':['#ffffff']
    , 'texts':

        np.random.choice(texts,300)
        
        #np.array(['12','13','14']).tolist()
        
    ,
    #'size_images':[((int(153/16)+1)*16 +2 ,18)] #for cnn_output_width = 19
    'size_images':[(max_length * 20 ,32)] #cnn_output_width * 16 + 2
    , 'is_simple_text':[True]
    , 'font_sizes':[12,14,16,18,22,23,24]
    , 'scale_size':[(None, 32)]
    , 'is_scale':[True]
},
#SQL white theme
{
    'fonts':fonts,
    'is_crop':[True]
    , 'text_colors':['#0505FE','#AFA4A4','#FD3D15','#FF00FF','#A59CA0','#030303']
    , 'background_colors':['#ffffff']
    , 'texts':

        np.random.choice(texts,300)
        
        #np.array(['12','13','14']).tolist()
        
    ,
    #'size_images':[((int(153/16)+1)*16 +2 ,18)] #for cnn_output_width = 19
    'size_images':[(max_length * 20 ,32)] #cnn_output_width * 16 + 2
    , 'is_simple_text':[True]
    , 'font_sizes':[12,14,16,18,22,23,24]
    , 'scale_size':[(None, 32)]
    , 'is_scale':[True]
}

]



if __name__ == "__main__":
    from MyGenerator.ImageGenerator import ImageGenerator
    it = ImageGenerator(train_conf)
    print(f'{it.size_word=}')

    # json_file = 'utils/dict.json'
    # array = []
    # with open(json_file, 'r' ) as f:
    #     di = json.load(f)
    #     array = [i for i in di if len(i) > 2 and len(i) < 50]
    #     print(len(array))
    #     for i in array[0:3]:
    #         print(i)

        
        