import os
import pickle 
import json
import re

base_dirs = [
    r'c:\work'
]
exclude_folder = [
    '.git','node_modules'
]

extentions = [
    '.cs','.sql'
]

ignore_word_paterns = [
    r'\d+',
    r'\d+-\d+-\d+',
    r'^\/\*+\/$',
    r'^\*+$',
    r'^\*+\/$',
    r'^\/\*+$'
]

path_result = 'dict.json'

def get_all_files(path):
    result_files = []
    for address, dirs, files in os.walk(path):
        for dir_name in exclude_folder:
            if dir_name in dirs:
                dirs.remove(dir_name)
        
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension in extentions:
                result_files.append(os.path.join(address,file))
        
    return result_files

def filter_word(word):
    word = word.strip().lower()
    if word == '': None    
    for patern in ignore_word_paterns:
        m = re.match(patern,word)
        if m is not None:
            return None
    return word

def get_words_from_file(file):
    words_result = set()
    with open(file,'r', encoding='UTF-8', errors='ignore') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '': continue
            words =  re.split('([ \.,])',line)
            for word in words:
                word = filter_word(word)
                if word is not None:                
                    words_result.add(word)
    return list(words_result)

def get_words():
    dictionary = {}
    max_length_word = 0
    for path in base_dirs:
        print(f'{path=}')
        for file in get_all_files(path):            
            #print(f'all files {len(file)}')
            words = get_words_from_file(file)
            #print(f'all words {len(words)}')
            for word in words:
                if len(word) > 200: continue
                if len(word) > max_length_word:
                    max_length_word = len(word)
                if word not in dictionary:
                    dictionary[word] = 1
                else:
                    dictionary[word] += 1
    print(f'{max_length_word=}')
    return dictionary
    
def save_dict(dict_word):
    dict_word_sorted = dict(sorted(dict_word.items(), key=lambda x:x[1],reverse = True))
    #print(dict_word_sorted)

    with open(path_result, 'w' ) as res:
        json.dump(dict_word_sorted, res, indent=4)
    print(f'saved to file {path_result=}')
    #with open('saved_dictionary.pkl', 'wb') as f:
        #pickle.dump(dictionary, f)
        
# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)


if __name__ == '__main__':
    #files = get_all_files(r'c:\work\KDB')
    #print(f'{len(files)}')
    #words = get_words_from_file(r'c:\work\KDB\Database\KDB\dbo\Stored Procedures\Finance_Invoice_Details_v6.sql')
    #print(words)
    #words = get_words_from_file('c:\\work\\Japps\\Apps\\ExternalUserTaskProviders\\Properties\\AssemblyInfo.cs')
    #print(words)

    dictionary = get_words()
    save_dict(dictionary)

    # print(filter_word('/***/'))
    # print(filter_word('***/'))
    # print(filter_word('/***'))
    

    