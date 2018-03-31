import scipy.io
import numpy as np

def load_mat(fname):
    data = scipy.io.loadmat(fname)
    X = data['X']
    y = data['y'].flatten()
    return X,y

def get_vocab_dict():
    words = {}
    inv_words = {}
    f = open('data/vocab.txt','r')
    for line in f:
        if line != '':
            (ind,word) = line.split('\t')
            words[int(ind)] = word.rstrip('\n')
            inv_words[word.rstrip('\n')] = int(ind)
    return words, inv_words
