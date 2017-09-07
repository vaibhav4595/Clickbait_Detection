from __future__ import division
import numpy as np
import pickle as pkl
import pdb
from ast import literal_eval
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Doc2Vec, doc2vec
import keras
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN, Dropout
from keras.layers.merge import concatenate, dot, multiply, add
from keras.models import Model
from keras.callbacks import ModelCheckpoint 
from attention import AttentionWithContext
import sys
import argparse
import codecs
import json
from extract_fc7 import create_embed

class Detector:
    """
    Creator Class for Clickbait Detection System
    """

    def __init__(self, title_max, word_embedding_size, doc2vec_size, image_size):
        """
        Initiliases the following parameters
        ===============================================
        * Maximum words to be considered in the title
        * Size of the word embeddings to be used
        * Embedding size of doc2vec
        ==============================================
        """

        self.model = None
        self.title_max = title_max
        self.word_embedding_size = word_embedding_size
        self.doc2vec_size = doc2vec_size
        self.image_size = image_size

    def set_params(self, activation):
        """
        Initiliases parameters for the model.

        Defines and sets the following parameters
        ===========================================================
        * Activation Function
        ===========================================================
        """

        self.activation = activation

    def create_model(self):
        """ 
        Initialises the input and layers for the model.
        =============================================================
        Model Component 1 (title word embeddings)
        * Input to this component is the word embeddings of the title.
          The length of the title is fixed to a particular value. Padding
          is done in case the title falls short, or truncated if the title becomes
          too long.
        * A BiLSTM layer with attention follows.
        * Sigmoid is then used to predict the class.

        Model Component 2 (title embeddings + document embedings)
        This is similar to the siamese approach of training.
        Both title and body are brought to the same vector space for comparison
        =============================================================
        """

        title_words = Input(shape=(self.title_max, self.word_embedding_size))
        text_embed_input = Input(shape=(self.doc2vec_size, ))
        title_embed_input = Input(shape=(self.doc2vec_size, ))
        image_embed_input = Input(shape=(self.image_size, ))
        image_small = Dense(300, activation=self.activation)(image_embed_input)

        # Layers for the Model Component 1
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(title_words)
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(lstm_layer)
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(lstm_layer)
        attention_layer = AttentionWithContext()(lstm_layer)
        dropout1 = Dropout(0.2)(attention_layer)
        left_hidden_layer1 = Dense(64, activation=self.activation)(dropout1)
        dropout2 = Dropout(0.2)(left_hidden_layer1)
        left_hidden_layer2 = Dense(32, activation=self.activation)(dropout2)

        # Layers for the Model Component 2 (weights are shared)
        shared_hidden_layer1 = Dense(128, activation=self.activation)
        text_hid1 = shared_hidden_layer1(text_embed_input)
        title_hid1 = shared_hidden_layer1(title_embed_input)

        shared_hidden_layer2 = Dense(64, activation=self.activation)
        text_hid2 = shared_hidden_layer2(text_hid1)
        title_hid2 = shared_hidden_layer2(title_hid1)

        shared_hidden_layer3 = Dense(32, activation=self.activation)
        text_hid3 = shared_hidden_layer3(text_hid2)
        title_hid3 = shared_hidden_layer3(title_hid2)

        elem_wise_vector = multiply([text_hid3, title_hid3])
        
        # Layers for the Model Component 3 (weights are shared)
        shared_hidden_layer_p1 = Dense(128, activation=self.activation)
        image_hid1 = shared_hidden_layer_p1(image_small)
        title_hid_p1 = shared_hidden_layer_p1(title_embed_input)

        shared_hidden_layer_p2 = Dense(64, activation=self.activation)
        image_hid2 = shared_hidden_layer_p2(image_hid1)
        title_hid_p2 = shared_hidden_layer_p2(title_hid_p1)

        shared_hidden_layer_p3 = Dense(32, activation=self.activation)
        image_hid3 = shared_hidden_layer_p3(image_hid2)
        title_hid_p3 = shared_hidden_layer_p3(title_hid_p2)

        elem_wise_vector2 = multiply([image_hid3, title_hid_p3])

        # Combines both the left and the right component
        combined1 = concatenate([left_hidden_layer2, elem_wise_vector, elem_wise_vector2])
        dropout_overall1 = Dropout(0.2)(combined1)
        combined2 = Dense(32, activation=self.activation)(dropout_overall1)

        # Predicts
        output = Dense(1, activation='sigmoid')(combined2)
#        output = Dense(1, activation='sigmoid')(elem_wise_vector2)

#        self.model = Model(inputs=[title_embed_input] + [image_embed_input], outputs=output)
        self.model = Model(inputs=[title_words] + [text_embed_input] + [title_embed_input] + [image_embed_input], outputs=output)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

        print self.model.summary()


    def fit_model(self, inputs, outputs, epochs):
        filepath="../weights/"+'3lstm'+"/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, outputs, validation_split=0.2, epochs=epochs, callbacks=callbacks_list, verbose=1)


def train():

    fp = open('../../clickbait17-validation-170630/instances.jsonl')
    all_lines = fp.readlines()
    lines = all_lines[:int(len(all_lines)*0.7)]
    
    fp2 = open('../../clickbait17-validation-170630/truth.jsonl')
    lines2 = fp2.readlines()
    
    article_embed = pkl.load(open('../data/article_embed.pkl'))
    image_embed = pkl.load(open('../data/image_embedding_4096.pkl'))

    words = []
    posts = []
    targets = []
    truth = []
    images = []
    max_len = 30

    truth_d = {}
    for line in tqdm(lines2):
        d = literal_eval(line)
        if d['truthClass'] == 'clickbait':
            truth_d[d['id']] = 1
        else:
            truth_d[d['id']] = 0

    word_vectors = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    for line in tqdm(lines):
        d = literal_eval(line)
        posts.append(article_embed['postText_'+d['id']])
        targets.append(article_embed['targetDescription_'+d['id']])
        truth.append(truth_d[d['id']])
        try:
            images.append(image_embed[d['id']])
        except:
            images.append(np.zeros(4096))
        text = d['postText'][0].split()
        l = len(text)
        temp = []
        if l >= max_len:
            for i in range(max_len):
                try:
                    temp.append(word_vectors[text[i]])
                except:
                    temp.append(np.zeros(300))

        else:
            pad = max_len - l
            for i in range(pad):
                temp.append(np.zeros(300))
            for i in range(l):
                try:
                    temp.append(word_vectors[text[i]])
                except:
                    temp.append(np.zeros(300))
        words.append(temp)

    words = np.array(words)
    posts = np.array(posts)
    targets = np.array(targets)
    images = np.array(images)

    tester = Detector(max_len, 300, 300, 4096)
    tester.set_params('relu')
    tester.create_model()
    tester.fit_model([words, posts, targets, images], truth, 10)
#    tester.fit_model([posts, images], truth, 10)


def test():

    model = Doc2Vec.load('../data/embed_model')
    fp = open('../../clickbait17-validation-170630/instances.jsonl')
    all_lines = fp.readlines()
    lines = all_lines[int(len(all_lines)*0.7):]
    
    fp2 = open('../../clickbait17-validation-170630/truth.jsonl')
    lines2 = fp2.readlines()
    
    article_embed = pkl.load(open('../data/article_embed.pkl'))
    image_embed = pkl.load(open('../data/image_embedding_4096.pkl'))

    words = []
    posts = []
    targets = []
    truth = []
    images = []
    max_len = 30

    truth_d = {}
    for line in tqdm(lines2):
        d = literal_eval(line)
        if d['truthClass'] == 'clickbait':
            truth_d[d['id']] = 1
        else:
            truth_d[d['id']] = 0
    
    word_vectors = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    for line in tqdm(lines):
        d = literal_eval(line)
        
        posts.append(model.infer_vector(d['postText']))
        targets.append(model.infer_vector(d['targetDescription']))
        text = d['postText'][0].split()
        truth.append(truth_d[d['id']])
        try:
            images.append(image_embed[d['id']])
        except:
            images.append(np.zeros(4096))
        l = len(text)
        temp = []
        if l >= max_len:
            for i in range(max_len):
                try:
                    temp.append(word_vectors[text[i]])
                except:
                    temp.append(np.zeros(300))

        else:
            pad = max_len - l
            for i in range(pad):
                temp.append(np.zeros(300))
            for i in range(l):
                try:
                    temp.append(word_vectors[text[i]])
                except:
                    temp.append(np.zeros(300))
        words.append(temp)

    words = np.array(words)
    posts = np.array(posts)
    images = np.array(images)
    targets = np.array(targets)
    
    tester = Detector(max_len, 300, 300, 4096)
    tester.set_params('relu')
    tester.create_model()
    tester.model.load_weights('../weights/3lstm/weights-04-0.39.hdf5')
    #out = tester.model.predict([posts, images])
    out = tester.model.predict([words, posts, targets, images])

    hit = 0
    for i in range(out.shape[0]):
        print out[i], truth[i]
        if out[i] >= 0.5 and truth[i] == 1:
            hit += 1
        elif out[i] <= 0.5 and truth[i] == 0:
            hit += 1

    print hit/out.shape[0]

def custom_test():

    samples = 30
    input1 = []
    for i in xrange(samples):
        input1.append(np.random.rand(5, 300))
    input1 = np.array(input1)

    input2 = []
    for i in xrange(samples):
        input2.append(np.random.rand(300, ))
    input2 = np.array(input2)

    input3 = []
    for i in xrange(samples):
        input3.append(np.random.rand(300, ))
    input3 = np.array(input3)

    output1 = np.random.randint(2, size=samples)

    tester = Detector(5, 300, 300)
    tester.set_params('relu')
    tester.create_model()
    tester.fit_model([input1, input2, input3], output1, 10)

def run_tira(input_path, result_path):

    print "creating image embeddings"
    create_embed(input_path)
    print "image embeddings done, loading doc2vec embeddings"
    model = Doc2Vec.load('/home/tuna/Clickbait_Detection/embed_model')
    fp = open(input_path+'/instances.jsonl')
    lines = fp.readlines()
    print "loading article embeddings"
    article_embed = pkl.load(open('/home/tuna/Clickbait_Detection/article_embed.pkl'))
    print "loading image embeddings generated previously"
    image_embed = pkl.load(open('/mnt/data/image-embeds/image_embed_4096.pkl'))

    words = []
    posts = []
    targets = []
    images = []
    ids = []
    max_len = 30
    
    res_file = codecs.open(result_path+'/results.jsonl', 'w+', encoding='utf-8')
    print "loading word2vec embeddings"
    word_vectors = KeyedVectors.load_word2vec_format('/home/tuna/Clickbait_Detection/GoogleNews-vectors-negative300.bin', binary=True)
    for line in tqdm(lines):
        d = literal_eval(line)
        ids.append(d['id'])
        posts.append(model.infer_vector(d['postText']))
        targets.append(model.infer_vector(d['targetDescription']))
        text = d['postText'][0].split()        
        try:
            images.append(image_embed[d['id']])
        except:
            images.append(np.zeros(4096))
        l = len(text)
        temp = []
        if l >= max_len:
            for i in range(max_len):
                try:
                    temp.append(word_vectors[text[i]])
                except:
                    temp.append(np.zeros(300))

        else:
            pad = max_len - l
            for i in range(pad):
                temp.append(np.zeros(300))
            for i in range(l):
                try:
                    temp.append(word_vectors[text[i]])
                except:
                    temp.append(np.zeros(300))
        words.append(temp)

    words = np.array(words)
    posts = np.array(posts)
    images = np.array(images)
    targets = np.array(targets)
    
    tester = Detector(max_len, 300, 300, 4096)
    tester.set_params('relu')
    tester.create_model()
    print "loading model weights"
    tester.model.load_weights('/home/tuna/Clickbait_Detection/weights-01-0.47.hdf5')
    out = tester.model.predict([words, posts, targets, images])
    for i in range(len(out)):
        res = {}
        res['id'] = ids[i]
        res['clickbaitScore'] = float(out.tolist()[i][0])
        res_file.write(json.dumps(res, ensure_ascii=False)+'\n')
    res_file.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Enter absolute path of input directory")
    parser.add_argument('-o', '--output', help="Enter absolute path of results directory")
    args = parser.parse_args()
    ip_path = args.input
    op_path = args.output
    run_tira(ip_path, op_path)
    # test()
