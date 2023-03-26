

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
            
words = get_words('data/onegin.txt')

import itertools
c = itertools.cycle(words)

configuration = [
# {
#     'fonts':['dataset/MyGenerator/font.ttf','dataset/MyGenerator/XO_Oriel_Bi.ttf']
#     , 'texts':[f'''{next(c)} {next(c)} {next(c)}
#                {next(c)} {next(c)}
#                {next(c)},
#                {next(c)} {next(c)} {next(c)}
               
#                {next(c)} 
#                {next(c)} {next(c)},{next(c)} {next(c)}
#                {next(c)},
#                {next(c)} {next(c)} {next(c)} {next(c)}
#                {next(c)} {next(c)}
#                {next(c)},
#                {next(c)} {next(c)} {next(c)} {next(c)}
#                ''',
#         f'''
               
               
               
#                {next(c)} {next(c)} {next(c)}
#                {next(c)} {next(c)}
#                {next(c)},
#                {next(c)} {next(c)} {next(c)} {next(c)}
#                {next(c)} {next(c)}
#                {next(c)},
#                {next(c)} {next(c)} {next(c)} {next(c)}
#                ''',f'''{next(c)} {next(c)} {next(c)}
#                {next(c)} {next(c)}
#                {next(c)},
#                {next(c)} {next(c)} {next(c)} {next(c)}
               
#                {next(c)} {next(c)} {next(c)}
#                {next(c)} {next(c)}
#                {next(c)},
#                {next(c)} {next(c)} {next(c)} {next(c)}
#                {next(c)} {next(c)}
#                {next(c)},
#                {next(c)} {next(c)} {next(c)} {next(c)}
#                ''']
#     , 'text_colors':['#0000ff','#00ff00','#ff0000','#ff00ff','#000000','#ffffff'] #,'#ff0000'
#     , 'size_images':[(300,300)]
#     , 'position_texts':[(5,12),(20,12),(50,12)] #,
#     , 'font_sizes':[12,15]
#     , 'background_colors':['#0000ff','#00ff00','#ff0000','#ff00ff','#000000','#ffffff']
#     , 'use_box':[False]
#     , 'box_colors':['#555555']
#     #, 'use_lines':['','|','-','\\','/']
#     , 'use_lines':['0000','0003','0002']
#     , 'lines_colors':['#555555']
#     , 'box_colors':['#555555']
#     # , 'use_special_chars':[True, False]
# },

#simple C# example black theme
{ 
    'fonts':['dataset/MyGenerator/font.ttf','dataset/MyGenerator/XO_Oriel_Bi.ttf']
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
    , 'size_images':[(400,400)]
    , 'position_texts':[(10,0),(10,100),(10,20)]
    , 'font_sizes':[12,15]
    , 'background_colors':['#1E1E1E']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0002','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
#simple class with field C# example black theme
{ 
    'fonts':['dataset/MyGenerator/font.ttf','dataset/MyGenerator/XO_Oriel_Bi.ttf']
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
'''
    ]
    , 'text_colors':['#D8D9D9']
    , 'size_images':[(400,400)]
    , 'position_texts':[(10,0),(10,20)]
    , 'font_sizes':[12,15]
    , 'background_colors':['#1E1E1E']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0003','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
#simple C# example white theme
{
    'fonts':['dataset/MyGenerator/font.ttf','dataset/MyGenerator/XO_Oriel_Bi.ttf']
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
    , 'size_images':[(400,400)]
    , 'position_texts':[(15,10),(5,50),(5,20)]
    , 'font_sizes':[15,15]
    
    , 'background_colors':['#ffffff']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0002','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
#simple SQL example white theme
{
    'fonts':['dataset/MyGenerator/font.ttf','dataset/MyGenerator/XO_Oriel_Bi.ttf']
    , 'texts':[
f'''<format color=#0505FE>select top</format> 10 <format color=#FF00FF>getdate</format><format color=#A59CA0>(), *</format>
<format color=#0505FE>from</format> staff_employee e
\t<format color=#AFA4A4>join</format> staff_person p <format color=#0505FE>on</format> e.personId = p.EmployeeId
<format color=#0505FE>where</format> e.DisplayName <format color=#AFA4A4>like</format> <format color=#FD3D15>'%akorobov%'</format>
'''
    ]
    , 'text_colors':['#030303']
    , 'size_images':[(400,400)]

    , 'position_texts':[(15,10),(5,50),(5,20),(15,120)]
    , 'font_sizes':[12,15]
    , 'background_colors':['#ffffff']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0002','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
#SQL many join example white theme
{
    'fonts':['dataset/MyGenerator/font.ttf','dataset/MyGenerator/XO_Oriel_Bi.ttf']
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
    , 'size_images':[(400,400)]
    , 'position_texts':[(15,10),(5,50),(5,20)]
    , 'font_sizes':[12,15]
    , 'background_colors':['#ffffff']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0002','0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
{ 
    'fonts':['dataset/MyGenerator/font.ttf','dataset/MyGenerator/XO_Oriel_Bi.ttf']
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
    , 'size_images':[(400,400)]
    , 'position_texts':[(10,0)]
    , 'font_sizes':[12]
    , 'background_colors':['#ffffff','#000000']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
}
]