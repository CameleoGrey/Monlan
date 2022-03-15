'''
Created on Dec 31, 2019

@author: bookknight
'''

from keras import backend as K
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Dense, Input, Lambda
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPool1D, Conv2D, MaxPooling2D, UpSampling1D, UpSampling2D, MaxPool2D
from keras.layers import Reshape, Conv2DTranspose
from sklearn.preprocessing import LabelBinarizer

import numpy as np
import joblib

# using Conv2D
class MyAutoencoder():

    def __init__(self, filters=100, kernel_size=3, latent_dim=100):

        self.filters = []
        self.divs = [1, 2]
        for div in self.divs:
            self.filters.append(filters // div)
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size

        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        pass

    def build_model(self, filters, feature_shape):

        # Encoder
        input = Input(feature_shape)
        x = input
        for n_filters in self.filters:
            x = Conv2D(filters=n_filters,
                       kernel_size=self.kernel_size,
                       activation="tanh",
                       strides=2,
                       padding="same")(x)
            x = MaxPool2D(padding="same")(x)
        shape = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(self.latent_dim)(x)
        self.encoder = Model(input, latent, name="encoder")
        self.encoder.summary()

        # Decoder
        decoder_input = Input(shape=(self.latent_dim,))
        x = Dense(shape[1] * shape[2] * shape[3])(decoder_input)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        self.filters = [128, 96, 80, 64]
        for filters in self.filters[::-1]:
            x = Conv2D(filters=filters,
                                kernel_size=self.kernel_size,
                                activation='tanh',
                                strides=1,
                                padding='same')(x)
            x = UpSampling2D()(x)
        decoder_output = Conv2DTranspose(filters=1,
                                         kernel_size=self.kernel_size,
                                         activation='tanh',
                                         padding='same',
                                         name='decoder_output')(x)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.autoencoder = Model(input, self.decoder(self.encoder(input)), name="autoencoder")
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.summary()

        pass

    def fit(self, x, y, batch_size=32, epochs=10, verbose=1):
        self.autoencoder.fit(x, y, batch_size=batch_size, verbose=verbose, epochs=epochs, shuffle=False)
        return self

    def predict(self, x):
        extractedFeatures = self.encoder.predict(x, batch_size=32)
        return extractedFeatures
