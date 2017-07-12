import numpy as np
import keras
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN, Dropout
from keras.layers.merge import concatenate, dot, multiply
from keras.models import Model
from attention import AttentionWithContext


class Detector:
    """
    Creator Class for Clickcbait Detection System
    """

    def __init__(self, title_max, word_embedding_size, doc2vec_size):
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

        # Layers for the Model Component 1
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(title_words)
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

        # Combines both the left and the right component
        combined1 = concatenate([left_hidden_layer2, elem_wise_vector])
        dropout_overall1 = Dropout(0.2)(combined1)
        combined2 = Dense(32, activation=self.activation)(dropout_overall1)

        # Predicts
        output = Dense(1, activation='sigmoid')(combined2)

        self.model = Model(inputs=[title_words] + [text_embed_input] + [title_embed_input], outputs=output)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

        print self.model.summary()


    def fit_model(self, inputs, outputs, epochs):
        self.model.fit(inputs, outputs, epochs=epochs, verbose=1)


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


if __name__=="__main__":
    custom_test()
