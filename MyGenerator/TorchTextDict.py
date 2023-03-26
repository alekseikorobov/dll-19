import torch
from typing import List
import numpy as np
import itertools

class TorchTextDict:
    def __init__(self, all_alph, word_size=100, counts_word = 200) -> None:
        self.word_size = word_size
        self.all_alph = all_alph
        self.counts_word = counts_word
        self.dict_chars = {
            'empt': 0
        }
        self.inds = {
            0: 'empt'
        }
        for w in all_alph:
            if w not in self.dict_chars:
                self.inds[len(self.dict_chars)] = w
                self.dict_chars[w] = len(self.dict_chars)
        
        self.tokenz = torch.zeros((0, self.word_size), dtype=torch.int32)

    def append_labels(self, words: List[str]):
        t = self.fit_transform(words)
        self.append_tokenz(t)

    def append_tokenz(self, tch: torch.LongTensor):
        self.tokenz = torch.cat([self.tokenz, tch])

    # def fit(self,words:List[str]):
    #     pass
    
    def get_labels(self, tch: torch.LongTensor) -> List[str]:
        '''
            shape = N, size
        '''
        words = []
        for t in tch:
            word = self.get_label(t)
            if word == '': continue
            words.append(word)
        return words

    def get_label(self, tch: torch.LongTensor) -> str:
        '''
            shape = size
        '''
        assert tch.dim() == 1, f'dim must be 1 now {tch.dim()}'
        t = list(itertools.takewhile(lambda index: index !=0, tch.numpy()))
        word = ''.join(
            map(lambda index: self.inds.get(index, '?'),t)
        )
        return word

    # Tensor shape - n1,n2
    #размер тензора должен быть всегда одинаковым, для этого размер количества слов сделаем параметром, иначе выходит ошибка:
    #stack expects each tensor to be equal size, but got [23, 100] at entry 0 and [17, 100] at entry 8
    def fit_transform_numpy(self, words: List[str]): # -> np.ndarray[np.int32]
        res = np.zeros((self.counts_word, self.word_size), dtype=np.int32)
        for idx, word in enumerate(words):
            if idx > self.counts_word: break
            chars = np.zeros(self.word_size, dtype=np.int32)

            for charIdx, w in enumerate(word):
                if charIdx > self.word_size: break
                if w not in self.dict_chars:
                    self.inds[len(self.dict_chars)] = w
                    self.dict_chars[w] = len(self.dict_chars)
                    # print(f'{self.dict_chars=}')
                    # print(f'{self.inds=}')

                chars[charIdx] = self.dict_chars[w]

            res[idx] = chars

        return res
    
    def fit_transform_list(self, words: List[str]) -> List[torch.Tensor]: # -> np.ndarray[np.int32]
        res = [0] * len(words)        
        for idx, word in enumerate(words):
            
            chars = self.fit_transform_word(word)

            res[idx] = chars

        return res

    def fit_transform_word(self, word) ->torch.Tensor:
        chars = torch.zeros(self.word_size, dtype=torch.int32)

        for charIdx, w in enumerate(word):
            if charIdx > self.word_size: break
            if w not in self.dict_chars:
                self.inds[len(self.dict_chars)] = w
                self.dict_chars[w] = len(self.dict_chars)
                    # print(f'{self.dict_chars=}')
                    # print(f'{self.inds=}')

            chars[charIdx] = self.dict_chars[w]
        return chars

    # Tensor shape - n1,n2
    def fit_transform(self, words: List[str]) -> torch.LongTensor:
        res = torch.zeros((len(words), self.word_size), dtype=torch.int32)
        for idx, word in enumerate(words):
            chars = torch.zeros(self.word_size, dtype=torch.int32)

            for charIdx, w in enumerate(word):
                if w not in self.dict_chars:
                    self.inds[len(self.dict_chars)] = w
                    self.dict_chars[w] = len(self.dict_chars)
                    # print(f'{self.dict_chars=}')
                    # print(f'{self.inds=}')

                chars[charIdx] = self.dict_chars[w]

            res[idx] = chars

        return res


# t = TorchTextDict()

# tch = t.fit_transform(['541'])

# t.append_labels(['123', '456'])

# t.append_labels(['678', '222'])

# print(f'{tch.shape=}')  # 2,3+97(empty)
# # print(f'{tch=}')

# l = t.get_labels(tch)
# print(l)

if __name__ == '__main__':
    t = TorchTextDict()
    res = t.fit_transform_list(['привет','пока'])
    print(f'{res=}')
    word = t.get_label(res[0])
    print(f'{word=}')