import numpy
import os
import sys

import numpy as np

from gensim.models.keyedvectors import KeyedVectors

file_path = "./GoogleNews-vectors-negative300.bin"

# return numpy array which is vector of the input word

def get_word_embeddings(word_vectors, input_word):
    vector = word_vectors[input_word]
    return vector

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
    if len(sys.argv) != 2:
        print 'Usage: python input_embeddings.py <word>'
        sys.exit(0)
    word_vectors = KeyedVectors.load_word2vec_format(file_path, binary=True)
    vector = get_word_embeddings(word_vectors, sys.argv[1])
    print vector