
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.regularizers import l2, l1_l2

class DenseAutoEncoder():
    """
    Class for creating autoencoder.
    """

    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        pass

    def buildModel(self, inputDim, latentDim, lr=0.001):

        encoder_in = Input(batch_input_shape=(None, inputDim, 1))
        x = Flatten()(encoder_in)
        x = Dense(256, activation="selu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        for i in range(4):
            x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Dropout(0.33)(x)
        x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        encoder_out = Dense(latentDim, activation="selu", activity_regularizer=l1_l2(l1=1e-6, l2=1e-6))(x)

        decoderIn = Input(shape=(latentDim,))
        x = Dense(128, activation="selu", kernel_initializer="he_normal")(decoderIn)
        x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        for i in range(4):
            x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Dropout(0.33)(x)
        x = Dense(256, activation="selu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        x = Dense(inputDim)(x)
        decoder_out = Reshape((inputDim, 1))(x)

        self.encoder = Model(encoder_in, encoder_out)
        self.decoder = Model(decoderIn, decoder_out)
        self.autoencoder = Model( self.encoder.input, self.decoder(self.encoder(encoder_in)) )

        self.autoencoder.compile(optimizer=Adam(lr), loss="mse")
        #self.encoder.summary()
        #self.autoencoder.summary()
        return self

    def fit(self, x, batchSize, epochs):
        self.autoencoder.fit(x, x, batch_size=batchSize, epochs=epochs)
        return self

    def encode(self, x, batchSize=32):
        feats = self.encoder.predict(x, batchSize)
        return feats

    def save_encoder(self, dir, name):
        print("saving feature extractor")
        feat_extr_json = self.encoder.to_json()
        with open(os.path.join(dir, "encoder_{}_architecture.json".format(name)), "w") as json_file:
            json_file.write(feat_extr_json)
        #self.encoder.save_weights("{}encoder_{}_weights.h5".format(dir, name))
        self.encoder.save_weights(os.path.join(dir,"encoder_{}_weights.h5".format(name)))
        pass

    def loadEncoder(self, dir, name):
        #json_file = open("{}encoder_{}_architecture.json".format(dir, name), "r")
        json_file = open(os.path.join(dir, "encoder_{}_architecture.json".format(name)), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        featExtr = model_from_json(loaded_model_json)

        # load weights into new model
        featExtr.load_weights(os.path.join(dir,"encoder_{}_weights.h5".format(name)))
        featExtr.compile(optimizer="adam", loss="mse")
        #featExtr.summary()
        return featExtr
