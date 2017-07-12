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

    def __init__(self, title_max, word_embedding_size):
        """
        Initiliases the following parameters
        ===============================================
        * Maximum words to be considered in the title
        * Size of the word embeddings to be used
        ==============================================
        """

        self.model = None
        self.title_max = title_max
        self.word_embedding_size = word_embedding_size

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
        Model Component 1
        * Input to this component is the word embeddings of the title.
          The length of the title is fixed to a particular value. Padding
          is done in case the title falls short, or truncated if the title becomes
          too long.
        * A BiLSTM layer with attention follows.
        * Sigmoid is then used to predict the class.
        =============================================================
        """

        title_words = Input(shape=(self.title_max, self.word_embedding_size))

        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(title_words)
        attention_layer = AttentionWithContext()(lstm_layer)
        dropout1 = Dropout(0.2)(attention_layer)
        hidden_layer1 = Dense(64, activation=self.activation)(dropout1)
        dropout2 = Dropout(0.2)(hidden_layer1)
        hidden_layer2 = Dense(32, activation=self.activation)(dropout2)

        output = Dense(1, activation='sigmoid')(hidden_layer2)

        self.model = Model(inputs=[title_words], outputs=output)
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

    output1 = np.random.randint(2, size=samples)

    tester = Detector(5, 300)
    tester.set_params('relu')
    tester.create_model()
    tester.fit_model(input1, output1, 10)


if __name__=="__main__":
    custom_test()
