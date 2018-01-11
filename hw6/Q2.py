# -*- coding: utf8 -*-
import jieba
import os, re
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from adjustText import adjust_text


def read_training_data(path):
    train_data = []
    with open(path, 'r') as data:
        lines = data.read().splitlines()
        for line in lines:
            train_data.append(line)
    print(len(train_data))
    return train_data

def cut(train_data):
    jieba.set_dictionary('./data/extra_dict/dict.txt.big.txt')

    output = open('./data/train_data_seg.txt', 'w')
    for line in train_data:
        temp = ''
        words = jieba.cut(line, cut_all=False)
        for word in words:
            output.write(str(word)+' ')
        output.write('\n')
    output.close()

def main():
    #read train data
    all_sents_path = os.path.join('.', 'data', 'all_sents.txt')

    train_data = read_training_data(all_sents_path)
    #cut sentece to words
    cut(train_data)

    #pretrain W2V
    sentences = word2vec.Text8Corpus('./data/train_data_seg.txt')

    model = word2vec.Word2Vec(sentences, size=200, workers = 6, min_count=10)

    print("vocabulary length: %d"%len(model.wv.vocab))
    model.save("./model/w2v_model_"+str(200)+"_"+str(10))
    model.wv.save_word2vec_format('./data/w2v', binary=False)

def filterWord(model):
    filter_vocabs = []
    filter_index = []
    for i, vocab in enumerate(model.wv.vocab):
        if model.wv.vocab[vocab] >= 3000 and model.wv.vocab[vocab] <= 6000:
             filter_vocabs.append(vocab)
             filter_index.append(i)
    return filter_index, filter_vocabs


def draw():
    print('load model')
    model = word2vec.Word2Vec.load('./model/w2v_model_200_10')
    vocab = []
    matplotlib.font_manager._rebuild()
    # print(plt.rcParams)
    for v in model.wv.vocab:
        if model.wv.vocab[v].count >= 3000:
            vocab.append(v)
    X = model[vocab]

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    font_name = "SimHei"
    plt.rcParams['font.family']=font_name
    plt.rcParams['axes.unicode_minus']=False

    xs, ys = X_tsne[:,0], X_tsne[:, 1]
    fig, ax = plt.subplots(figsize=(8, 8))
    texts = []
    for x, y, vocab in zip(xs, ys, vocab):
        ax.plot(x, y, '.')
        texts.append(plt.text(x, y, vocab))
    adjust_text(texts, arrowprops=dict(arrowstyle='-'))

    plt.savefig('./w.png')

def dispFonts():
    from matplotlib.font_manager import FontManager
    import subprocess

    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)
    print(mat_fonts)
    output = subprocess.check_output(
        'fc-list :lang=zh -f "%{family}\n"', shell=True)
    output = output.decode('utf-8')
    zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
    available = mat_fonts & zh_fonts

    print('*' * 10 +  u'可用的中文本體'+'*' * 10)
    for f in available:
        print(f)

if __name__ == '__main__':
    # main()
    draw()
    # dispFonts()
