from config_rec import all_alph
import numpy as np

def get_words(path,size=(4,5),count=100):
    res = set()
    with open(path,'r') as f:
        for line in f.readlines():
            text = line.strip().strip('\t').replace('\n','').lower()            
            if len(text)< size[0]:
                continue
            for word in text.split():
                if len(word)>=size[0] and len(word)<=size[1]:
                    res.add(word)
                    
                if len(res)>=count:
                    return list(res)            
    return list(res)
            
words = ['1','2']# get_words('example/TextBPNPlusPlus/data/onegin.txt')

import itertools
c = itertools.cycle(words)
def get_random_words(count_words=100,size_word=(1,8) ):
    rnd_length_list = np.random.randint(size_word[0],size_word[1],size=count_words)
    words = []    
    for length_word in rnd_length_list:        
        rnd_index_chares = np.random.randint(0,len(all_alph),size=length_word )        
        word = ''.join([all_alph[int(i)] for i in rnd_index_chares])
        words.append(word)
    return words
    

def prop_exists(word, p_exists, p_tab=0.1):
    p_ = np.random.randint(1,100)
    p_tab_ = np.random.randint(1,100)
    if p_tab_ < p_tab*100:
        word = '\t' + word
    if p_ < p_exists*100:
        return word
    return ' '*len(word)
        
    

re = '\n'
def get_random_text(rows = 22, max_length = 30, p_exists=0.7, p_tab=0.7):
    
    rand_list = get_random_words(count_words=300)
    result = []
    it = 0
    for row in range(rows):
        current_row = ''
        current_lenngth = 0
        while True:
            current_lenngth = len(current_row.replace('\t','    '))
            if current_lenngth > max_length or it>len(rand_list)-1:
                break
            current_row += prop_exists(rand_list[it] + ' ', p_exists, p_tab)
            it += 1
        
        result.append(current_row)    
    
    
    
    # np.random.choice([]) 
    
    return re.join(result)


fonts = ['example/TextBPNPlusPlus/dataset/MyGenerator/font.ttf'
         ,'example/TextBPNPlusPlus/dataset/MyGenerator/XO_Oriel_Bi.ttf'
         ,'example/TextBPNPlusPlus/dataset/MyGenerator/consolas/consola.ttf']

configuration = [
{
    'fonts':fonts
    , 'texts':[f'''{next(c)} {next(c)} {next(c)}
               {next(c)} {next(c)}
               {next(c)},
               {next(c)} {next(c)} {next(c)}
               
               {next(c)} 
               {next(c)} {next(c)},{next(c)} {next(c)}
               {next(c)},
               {next(c)} {next(c)} {next(c)} {next(c)}
               {next(c)} {next(c)}
               {next(c)},
               {next(c)} {next(c)} {next(c)} {next(c)}
               ''',
        f'''
               {next(c)} {next(c)} {next(c)}
               {next(c)} {next(c)}
               {next(c)},
               {next(c)} {next(c)} {next(c)} {next(c)}
               {next(c)} {next(c)}
               {next(c)},
               {next(c)} {next(c)} {next(c)} {next(c)}
               ''',f'''{next(c)} {next(c)} {next(c)}
               {next(c)} {next(c)}
               {next(c)},
               {next(c)} {next(c)} {next(c)} {next(c)}
               
               {next(c)} {next(c)} {next(c)}
               {next(c)} {next(c)}
               {next(c)},
               {next(c)} {next(c)} {next(c)} {next(c)}
               {next(c)} {next(c)}
               {next(c)},
               {next(c)} {next(c)} {next(c)} {next(c)}
               ''']
    , 'text_colors':['#0000ff','#00ff00','#ff0000','#ff00ff','#000000','#ffffff'] #,'#ff0000'
    , 'size_images':[(500,500)]
    , 'position_texts':[(5,12),(20,12),(50,12)] #,
    , 'font_sizes':[12,15]
    , 'background_colors':['#0000ff','#00ff00','#ff0000','#ff00ff','#000000','#ffffff']
    , 'use_box':[False]
    , 'box_colors':['#555555']
    #, 'use_lines':['','|','-','\\','/']
    , 'use_lines':['0000','0003','0002']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']
    # , 'use_special_chars':[True, False]
},

#simple C# example black theme
{ 
    'fonts':fonts
    , 'texts':[
f'''
<format color=#2C6DBF>using</format> System;
<format color=#2C6DBF>namespace</format> MyNameSpace;

<format color=#2C6DBF>public class</format> <format color=#4CC5AC>MyClass</format> 
{{
\t<format color=#2C6DBF>public string</format> Prop {{ <format color=#2C6DBF>get</format>; <format color=#2C6DBF>set</format>; }}
}}
'''
    ]
    , 'text_colors':['#D8D9D9']
    , 'size_images':[(500,500)]
    , 'position_texts':[(10,0),(10,100),(10,20),(20,200)]
    , 'font_sizes':[12,15,20,22]
    , 'background_colors':['#1E1E1E']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0002','0003','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
#simple class with field C# example black theme
{ 
    'fonts':fonts
    , 'texts':[
f'''
<format color=#2C6DBF>using</format> System;
<format color=#2C6DBF>namespace</format> MyNameSpace;

<format color=#2C6DBF>public class</format> <format color=#4CC5AC>MyClass</format> 
{{
\t<format color=#2C6DBF>public string</format> Prop {{ <format color=#2C6DBF>get</format>; <format color=#2C6DBF>set</format>; }}

\t<format color=#2C6DBF>private string</format> _prop1;
\t<format color=#2C6DBF>public string</format> Prop1
\t{{
\t\t<format color=#2C6DBF>get</format> {{ <format color=#9008CE>return</format> _prop1; }}
\t\t<format color=#2C6DBF>set</format> {{ _prop1 = <format color=#2C6DBF>value</format>; }}
\t}}
}}
''',
f'''
<format color=#2C6DBF>using</format> System;
<format color=#2C6DBF>namespace</format> MyNameSpace;

{{
<format color=#2C6DBF>public string</format> Prop {{ <format color=#2C6DBF>get</format>; <format color=#2C6DBF>set</format>; }}
\t<format color=#2C6DBF>public class</format> <format color=#4CC5AC>FDSFFSDFS</format> 
\t<format color=#2C6DBF>public string</format> Prop1
\t{{

\t\t<format color=#2C6DBF>sdfdsfs</format> {{ <format color=#9008CE>return</format> _prop1; }}
\t\t<format color=#2C6DBF>sdfdsfs</format> {{ 23sdfsdf = <format color=#2C6DBF>value</format>; }}
\t<format color=#2C6DBF>private string</format> _prop1;
\t}}
}}
'''
    ]
    , 'text_colors':['#D8D9D9']
    , 'size_images':[(500,500)]
    , 'position_texts':[(10,0),(10,20),(20,60)]
    , 'font_sizes':[12,15,22]
    , 'background_colors':['#1E1E1E']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0001','0003','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
{ 
    'fonts':fonts
    , 'texts':[
f'''
<format color=#7b7b7b size=10>0 references | 0 changes | 0 authors, 0 changes | 0 exception, - live</format>
<format color=#3077bb>public class</format> <format color=#4CC5AC>MyClass</format>: <format color=#4CC5AC>BaseClass</format>
{{
\t<format color=#3077bb>public</format> <format color=#86c691>DateTime</format> createDate;
\t<format color=#3077bb>protected string</format> login;
\t<format color=#3077bb>protected string</format> login1111;
\t<format color=#3077bb>protected string</format> login1111;
\t<format color=#3077bb>public</format> <format color=#86c691>DateTime</format> modifyDate;
\t<format color=#3077bb>protected string</format> login1111;

\t<format color=#7b7b7b size=10>0 references | 0 changes | 0 authors, 0 changes | 0 exception, - live</format>
\t<format color=#3077bb>public string</format> Prop111 {{ <format color=#3077bb>get</format>; <format color=#3077bb>set</format>; }}

\t<format color=#3077bb>private string</format> _prop1;

\t<format color=#7b7b7b size=10>0 references | 0 changes | 0 authors, 0 changes | 0 exception, - live</format>
\t<format color=#3077bb>public string</format> Prop1
\t{{
\t\t<format color=#3077bb>get</format> {{ <format color=#d8a0df>return</format> _prop1; }}
\t\t<format color=#3077bb>set</format> {{ _prop1 = <format color=#3077bb>value</format>; }}
\t}}
}}
'''
    ]
    , 'text_colors':['#D8D9D9']
    , 'size_images':[(500,500)]
    , 'position_texts':[(10,0),(10,10)]
    , 'font_sizes':[15]
    , 'background_colors':['#1E1E1E']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0000','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},

#simple C# example white theme
{
    'fonts':fonts
    , 'texts':[
f'''
<format color=#038003>// Online C# Editor for free</format>
<format color=#038003>// Write, Edit and Run your C#</format>

<format color=#2d01fd>using</format> System;

<format color=#2d01fd>public class</format> <format color=#2b9ec9>HelloWorld</format>
{{
\t<format color=#2d01fd>public static void</format> Main(<format color=#2d01fd>string</format>[] args)
\t{{
\t\t<format color=#2b9ec9>Console</format>.WriteLine (<format color=#b62a15>"Hello Mono World"</format>);
\t}}
}}
'''
    ]
    , 'text_colors':['#030303']
    , 'size_images':[(500,500)]
    , 'position_texts':[(15,10),(5,50),(5,20),(15,120)]
    , 'font_sizes':[15,18]
    
    , 'background_colors':['#ffffff']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0002','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
#simple SQL example white theme
{
    'fonts':fonts
    , 'texts':[
f'''<format color=#0505FE>select top</format> 10 <format color=#FF00FF>getdate</format><format color=#A59CA0>(), *</format>
<format color=#0505FE>from</format> staff_employee e
\t<format color=#AFA4A4>join</format> staff_person p <format color=#0505FE>on</format> e.personId = p.EmployeeId
<format color=#0505FE>where</format> e.DisplayName <format color=#AFA4A4>like</format> <format color=#FD3D15>'%akorobov%'</format>
'''
    ]
    , 'text_colors':['#030303']
    , 'size_images':[(500,500)]

    , 'position_texts':[(15,10),(5,50),(15,120),(15,220),(15,420)]
    , 'font_sizes':[12,13,15,16]
    , 'background_colors':['#ffffff']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0002','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
#SQL many join example white theme
{
    'fonts':fonts
    , 'texts':[
f'''<format color=#0505FE>select</format>
\te.Tepsd1,
\tsdfdsf = t.stst <format color=#AFA4A4>+</format> sdf,
\tf.54654,
\tp.cvbccvb,
<format color=#0505FE>from</format> staff_employee e
\t<format color=#AFA4A4>join</format> staff_person p <format color=#0505FE>on</format> e.personId = p.EmployeeId
\t<format color=#AFA4A4>join</format> firm f <format color=#0505FE>on</format> e.rrr = p.sdfds
\t<format color=#AFA4A4>join</format> sdgfsdfsdf p <format color=#0505FE>on</format> e.dfd = p.df
\t<format color=#AFA4A4>join</format> yuiuyi p <format color=#0505FE>on</format> e.mbn = p.yu
\t<format color=#AFA4A4>join</format> xcvxcv p <format color=#0505FE>on</format> e.fh = p.fgh
<format color=#0505FE>where</format> e.DisplayName <format color=#AFA4A4>like</format> <format color=#FD3D15>'%akorobov%'</format>
'''
    ]
    , 'text_colors':['#030303']
    , 'size_images':[(500,500)]
    , 'position_texts':[(15,10),(5,50),(5,120),(25,250)]
    , 'font_sizes':[12,15]
    , 'background_colors':['#ffffff']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0002','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
{ 
    'fonts':fonts
    , 'texts':[
f'''
Генри Ли III (англ. Henry Lee III; 29 января
1756 — 25 марта 1818) — американский военный
, участник Войны за независимость США, 
делегат Конгресса Конфедерации от Виргинии,
впоследствии 9-й губернатор штата Виргиния
и депутат Палаты представителей США 
от Виргинии, отец генерала армии Конфедерации
Роберта Ли.

В годы войны Ли служил офицером кавалерии 
в Континентальной армии, командовал 
специальным отрядом лёгкой кавалерии и был
известен под 
именем «Легкоконный Гарри». Он занимался в
охотой за фуражирами противника, но в 1779
осуществил нападение на форт Паулус-Хук
''',
f'''
Сразу после смерти Александра возник вопрос
относительно престолонаследника. Особенность
передачи власти заключалась в том, что никто
из реальных претендентов на царский престол
физически не мог управлять громадной империей
и в случае избрания требовал опеки. 
Военачальники
рассматривали три кандидатуры — малолетнего
сына Александра от Барсины Геракла, ребёнка
беременной Роксаны, в случае если родится
мальчик, и слабоумного единокровного брата
Арридея. После короткого периода вооружённого
противостояния, в ходе которого был убит один
из претендентов на роль регента Мелеагр,
реальную власть получил Пердикка.
'''
,f'''
1152 — Фридрих I Барбаросса занял престол
1789 — вступила в силу Конституция США;
1803 — в России вышел «Указ о вольных
1818 — в Москве открыт бронзовый памятник
1877 — в Большом театре впервые поставлен
1890 — в Эдинбурге открыто движение по мосту
1919 — в Москве принято решение об учреждении
1936 — свой первый полёт совершил дирижабль
В результате лобового столкновения пассажирского
Лионель Месси и Алексия Путельяс признаны 
По меньшей мере 63 человека погибли в 
Главный приз 73-го Берлинского международного
'''
    ]
    , 'text_colors':['#000000','#ffffff']
    , 'size_images':[(500,500)]
    , 'position_texts':[(10,0)]
    , 'font_sizes':[12,15]
    , 'background_colors':['#ffffff','#000000']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
{ 
    'fonts':fonts
    , 'texts':[
'''
\t\t\t()
{
\t{
\t\t\t\t\t{ } \t \t { }
\t}
\t\t\t\t\t\t\t [  ]    [    ]
}
\t\t\t\t}
\t\t\t\t [  ]    [    ]

''',
'''
\t\t\t()
{   \t  \t * * *
\t{ *
\t\t\t\t\t{ } \t \t { } *
\t} *
\t\t\t\t\t\t\t [  ]    [    ]   *
}   *
\t\t\t\t}   *
\t\t\t\t [  ]    [    ] *

'''
,'''
\t\t\t()
{
\t{
\t\t\t\t\t{ } \t \t { }
\t;}
\t\t\tt\t\t [ 123213 ]    [    ;}
};
\t\t\t\t}  ;;;
\t\t\t\t [  ;}    [dsfdsfs;}

''',
''';;;
\t\t\t(;;)
{   \t  \t * * *
\t{ *
\t\t\\t{ } \t \t { } *
\t} *
\t\t\t\t\t\t\t [  ]    [    ]   *
}   *
\t\t\t\t}   *       ;;;;;
\t\t\t\t [  ]    [    ] *

'''
    ]
    , 'text_colors':['#000000','#ffffff']
    , 'size_images':[(500,500)]
    , 'position_texts':[(20,10),(20,50),(10,100)]
    , 'font_sizes':[12,15,17]
    , 'background_colors':['#ffffff','#000000']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0000','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},

{ 
    'fonts':['example/TextBPNPlusPlus/dataset/MyGenerator/font.ttf']
    , 'texts':[
# f''' {''.join(map(str,range(30)))}
#      {re.join(map(str,range(28)))}
# '''
get_random_text(rows = 22, max_length = 40, p_exists=0.7,p_tab=.5),
get_random_text(rows = 22, max_length = 40, p_exists=0.8,p_tab=1),
get_random_text(rows = 22, max_length = 40, p_exists=0.2,p_tab=0.2),
get_random_text(rows = 22, max_length = 40, p_exists=0.7,p_tab=.5),
get_random_text(rows = 22, max_length = 40, p_exists=0.8,p_tab=1),
get_random_text(rows = 22, max_length = 40, p_exists=0.2,p_tab=0.2),


    ]
    , 'text_colors':['#000000','#ffffff']
    , 'size_images':[(500,500)]
    , 'position_texts':[(4,0)]
    , 'font_sizes':[17]
    , 'background_colors':['#ffffff','#000000']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0000','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
}
]