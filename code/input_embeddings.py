#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy
import os
import sys
import json
import codecs

import numpy as np
import cPickle as pickle

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from gensim.models.keyedvectors import KeyedVectors
from collections import OrderedDict

file_path = "./GoogleNews-vectors-negative300.bin"

data_path = './data/'
truth_file = data_path+'truth.jsonl'
instances_file = data_path+'instances.jsonl'

filters = 20
kernel_size = 2
epochs = 200
maxlen = 20

char_vec_file = open('./data/character_vectors.pkl', 'rb')
init_vectors = pickle.load(char_vec_file)
char_vec_file.close()

# return numpy array which is vector of the input word

def get_word_embeddings(word_vectors, input_word):
    vector = word_vectors[input_word]
    return vector

# full_text is the all the text from which you want character vectors
# making a one hot encoded vector for each character
def init_character_vectors():
    f1 = codecs.open(instances_file, 'r', 'utf-8')
    full_text = f1.read()
    f1.close()
    characters = list(OrderedDict.fromkeys(full_text).keys())
    initial_vectors = {}
    i=0
    for char in characters:
        if i == len(characters):
            break
        initial_vectors[char] = np.zeros(shape=(1, len(characters)), dtype=float)
        initial_vectors[char][0, i] = 1
        i=i+1
    f = open('./data/character_vectors.pkl', 'wb')
    pickle.dump(initial_vectors, f)
    f.close()

# character for which vector is needed
def get_character_vectors(char):
    return init_vectors[char]

def word_representation(word):
    rep = []
    for x in word:
        rep.append(get_character_vectors(x))
    network_input = np.array(rep)
    return network_input

def char_embed_layered_archi():
    # 1. create vector for each letter in the corpus
    # 2. 3 layers of 1-dimensional CNN with ReLU non-linearity on each vector of character sequence of that word
    # 3. max-pooling across the sequence for each convolutional feature

    network = Sequential()
    network.add(Conv1D(12, kernel_size, padding='valid', activation='relu', strides=1))
    network.add(Conv1D(7, kernel_size, padding='valid', activation='relu', strides=1))
    network.add(Conv1D(3, kernel_size, padding='valid', activation='relu', strides=1))
    network.add(GlobalMaxPooling1D(padding='valid'))
    network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath="./data/char-embed-model-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return network

def generate_char_embeddings():
    # pad each word before processing
    # x will be single char vectors of the word
    # y will be the word itself
    x = []
    y = []
    with open(instances_file, 'r') as instances:
        for line in instances:
            t = json.loads(line)
            keywords = t['targetKeywords']
            post_text = t['postText']
            target_description = t['targetDescription']
            for keyword in keywords:
                x.append(word_representation(keyword))
                y.append(keyword)
            for word in post_text:
                x.append(word_representation(word))
                y.append(word)
            for word in keywords:
                x.append(word_representation(word))
                y.append(word)
    network = char_embed_layered_archi()
    network.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.33, batch_size=10)
    network.save('/data/char-embed-network')

# https://github.com/minimaxir/char-embeddings/blob/master/create_embeddings.py

def averaged_out_char_embeddings_from_word():
    vectors = {}
    with open(file_path, 'rb') as f:
        for line in f:
            line_split = line.strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            word = line_split[0]
            for char in word:
                if ord(char) < 128:
                    if char in vectors:
                        vectors[char] = (vectors[char][0] + vec, vectors[char][1] + 1)
                    else:
                        vectors[char] = (vec, 1)

    base_name = os.path.splitext(os.path.basename(file_path))[0] + '-char.txt'
    with open(base_name, 'wb') as f2:
        for word in vectors:
            avg_vector = np.round((vectors[word][0] / vectors[word][1]), 6).tolist()
            f2.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")

if __name__ == '__main__':
    init_character_vectors()
    # if len(sys.argv) != 2:
    #     print 'Usage: python input_embeddings.py <word>'
    #     sys.exit(0)
    # word_vectors = KeyedVectors.load_word2vec_format(file_path, binary=True)
    # vector = get_word_embeddings(word_vectors, sys.argv[1])