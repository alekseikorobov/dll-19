

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

fonts = ['MyGenerator/font.ttf'
        # ,'MyGenerator/XO_Oriel_Bi.ttf'
        # ,'MyGenerator/consolas/consola.ttf'
         ]

configuration = [
{ 
    'fonts':fonts
    , 'texts':[
f'''
<format color=#2C6DBF>using</format> System;

<format color=#2C6DBF>public class</format> <format color=#4CC5AC>MyClass</format> 
<format color=#2C6DBF>namespace</format> MyNameSpace;
{{
\t<format color=#2C6DBF>public string</format> Prop {{ <format color=#2C6DBF>get</format>; <format color=#2C6DBF>set</format>; }}
}}
'''
    ]
    , 'text_colors':['#D8D9D9']
    , 'size_images':[(400,400)]
    , 'position_texts':[(10,50)]
    , 'font_sizes':[14]
    , 'background_colors':['#1E1E1E']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0000']
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
\t<format color=#2C6DBF>public string</format> Prop1
\t{{
\t\t<format color=#2C6DBF>get</format> {{ <format color=#9008CE>return</format> _prop1; }}
\t\t<format color=#2C6DBF>set</format> {{ _prop1 = <format color=#2C6DBF>value</format>; }}
\t}}
\t<format color=#2C6DBF>private string</format> _prop1;

}}
'''
    ]
    , 'text_colors':['#D8D9D9']
    , 'size_images':[(400,400)]
    , 'position_texts':[(8,5)]
    , 'font_sizes':[17]
    , 'background_colors':['#1a1a1a']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
#simple C# example white theme
{
    'fonts':fonts
    , 'texts':[
f'''
<format color=#2d01fd>public class</format> <format color=#2b9ec9>HelloWorld</format>
{{
\t<format color=#2d01fd>public static void</format> Main(<format color=#2d01fd>string</format>[] args)
\t{{
\t\t<format color=#2b9ec9>Console</format>.WriteLine (<format color=#b62a15>"Hello Mono World"</format>);
\t}}
<format color=#038003>// Online C# Editor for free</format>
}}

<format color=#038003>// Write, Edit and Run your C#</format>

<format color=#2d01fd>using</format> System;

'''
    ]
    , 'text_colors':['#030303']
    , 'size_images':[(400,400)]
    , 'position_texts':[(15,10)]
    , 'font_sizes':[14]
    
    , 'background_colors':['#ffffff']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0002']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
#simple SQL example white theme
{
    'fonts':fonts
    , 'texts':[
f'''
<format color=#0505FE>from</format> staff_employee e
\t<format color=#AFA4A4>join</format> staff_person p <format color=#0505FE>on</format> e.personId = p.EmployeeId
<format color=#0505FE>where</format> e.DisplayName <format color=#AFA4A4>like</format> <format color=#FD3D15>'%akorobov%'</format>
<format color=#0505FE>select top</format> 10 <format color=#FF00FF>getdate</format><format color=#A59CA0>(), *</format>
<format color=#0505FE>from</format> staff_employee e
\t<format color=#AFA4A4>join</format> staff_person p <format color=#0505FE>on</format> e.personId = p.EmployeeId
<format color=#0505FE>where</format> e.DisplayName <format color=#AFA4A4>like</format> <format color=#FD3D15>'%akorobov%'</format>
<format color=#0505FE>select top</format> 10 <format color=#FF00FF>getdate</format><format color=#A59CA0>(), *</format>
'''
    ]
    , 'text_colors':['#030303']
    , 'size_images':[(400,400)]

    , 'position_texts':[(10,8)]
    , 'font_sizes':[16]
    , 'background_colors':['#ffffff']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
},
{ 
    'fonts':fonts
    , 'texts':[
f'''
рассматривали три кандидатуры — малолетнего
сына Александра от Барсины Геракла, ребёнка
беременной Роксаны, в случае если родится
мальчик, и слабоумного единокровного брата
Арридея. После короткого периода вооружённого
противостояния, в ходе которого был убит один
из претендентов на роль регента Мелеагр,
реальную власть получил Пердикка.
Сразу после смерти Александра возник вопрос
относительно престолонаследника. Особенность
передачи власти заключалась в том, что никто
из реальных претендентов на царский престол
физически не мог управлять громадной империей
и в случае избрания требовал опеки. 
Военачальники

'''
    ]
    , 'text_colors':['#333333']
    , 'size_images':[(400,400)]
    , 'position_texts':[(10,10)]
    , 'font_sizes':[12]
    , 'background_colors':['#fffffa']
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
\t\t\t{
\t\t\t\t{
\t}
\t\t\t\t\t{ } \t \t { }
}
\t\t\t\t [  ]    [    ]
\t\t\t\t\t\t\t [  ]    [    ]

\t\t\t\t}
''',
    ]
    , 'text_colors':['#ffffff']
    , 'size_images':[(400,400)]
    , 'position_texts':[(20,10)]
    , 'font_sizes':[16]
    , 'background_colors':['#000000']
    , 'use_box':[False]#not word
    , 'box_colors':['#555555']
    , 'use_lines':['0000']
    , 'lines_colors':['#555555']
    , 'box_colors':['#555555']#not word
}              
]