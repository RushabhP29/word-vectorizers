import gensim
import rope
import os

from gensim.models import Word2Vec


class MySentences(object):
    def __init__(self,dirname):
        self.dirname =dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname,fname)):
                yield line.split()


print('')
print('>>Start......')
print('')
wv_size = 100
wv_window = 4
wv_min_count = 10
wv_sg =1
wv_workers = 4

sentences = MySentences('data')
model = Word2Vec(sentences,size=wv_size,window=wv_window,min_count=wv_min_count,sg=wv_sg,workers=wv_workers)
model.save('bin')
print('>>Model saved')
print('')
