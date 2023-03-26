

from typing import List,Tuple

import re

class Format:
    def __init__(self,color=None,size=None) -> None:
        self.color = color
        self.size = size
        self.is_space = False
        self.pix_size_width = 0
        self.pix_size_height = 0


    def __str__(self) -> str:
        return f'color:{self.color}, size:{self.size}, is_space:{self.is_space}, pix_size_w:{self.pix_size_width}'
    
    def __repr__(self) -> str:
        return f'color:{self.color}, size:{self.size}, is_space:{self.is_space}, pix_size_w:{self.pix_size_width}'

regex_split = r"(\w+=[\#0-9abcdef]+)"
def parse_format(params:str) ->Format:
    f = Format()
    params = params.lower().strip()
    #print(f'{params=}')
    splits = params.split(' ')
    #print(f'{splits=}')
    for s in splits:
        s  = s.strip()
        if s == '': continue
        
        splits1 = re.split(r"=", s)
        #print(f'{splits1=}')
        
        key = splits1[0].strip()
        value = splits1[1].strip()
        
        f.__dict__[key] = value


    return f

regex_split = r"( {0,}<format.*?>.*?<\/format>)"
regex_match = r" {0,}<format(.*?)>(.*?)<\/format>"

def parse_text_format(text_line)->List[Tuple[str,Format]]:
    '''
        return list of text, and format
    '''
    texts = []

    
    splits = re.split(regex_split, text_line)
    #print(f'{splits=}')

    for sp in splits:
        if sp.strip(' ') == '': continue
        m = re.match(regex_match, sp)
        if m is not None:
            patern, text = m.groups()
            format = parse_format(patern)
            if len(texts) > 0 and re.match(f'^ +.*',sp):
                format.is_space = True
            for idx,word in enumerate(text.split(' ')):
                if idx > 0 and len(texts)>0:
                    format1 = Format()
                    format1.is_space = True
                    format1.color = format.color
                    format1.size = format.size
                    format = format1
                if word == '': continue
                texts.append([word,format])
        else:
            ft = Format()
            for idx, word in enumerate(sp.split(' ')):
                #print(f'{idx, word=}')
                if idx > 0 and len(texts)>0:
                    ft = Format()
                    ft.is_space = True                    
                if word == '': 
                    continue
                texts.append([word,ft])

    return texts




if __name__ == '__main__':
    #print(f'''{parse_format('color=#0000ff')}''')
    #print(f'''{parse_format('   color   =   #0000ff   ')}''')
    #print(f'''{parse_format('size=15')}''')
    #print(f'''{parse_format('size=15 color=#0000ff')}''')
    #print(f'''{parse_format('    size =     15        color   =    #0000ff    ')}''')

    # print(f'''{parse_text_format('1 ')}''') #false
    # print(f'''{parse_text_format(' 1')}''') #false
    # print(f'''{parse_text_format(' 1 ')}''') #false
    #print(f'''{parse_text_format('1 1 1')}''') #false true true

    #V print(f'''{parse_text_format('1 <format color=#2C6DBF>using</format>')}''') #false true
    #Vprint(f'''{parse_text_format('1<format color=#2C6DBF>using</format>')}''') #false false
    #Vprint(f'''{parse_text_format('<format color=#2C6DBF>using1</format> <format color=#2C6DBF>using2</format>')}''') #false true
    #Vprint(f'''{parse_text_format('<format color=#2C6DBF>using</format> System')}''') #false true
    #Vprint(f'''{parse_text_format('<format color=#2C6DBF>us ing</format>')}''') #false true


    text = "<format color=#000000 size=10>0</format>"
    print(f'''{parse_text_format(text)}''') #false true

    #print(f'''{parse_text_color('ttt1 tt2t <format size=15 color=#0000ff>clrfass</format>  <format color=#0000ff>cla4ss</format>, My2Cla ss')}''')
    #print(f'''{parse_text_color('text')}''')
    #print(f'''{parse_text_color('text <format color=#ffffff>Class</format> text text')}''')
    #print(f'''{parse_text_color('<format color = #ffffff >Class</format> text')}''')
    #print(f'''{parse_text_color('text <format color=#ffffff>Class</format> text')}''')
    #print(f'''{parse_text_color('<format color=#ffffff size=12>Class</format> <format color=#ffffff>Class</format>')}''')
    pass
    
