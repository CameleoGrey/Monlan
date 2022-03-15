

from keras.models import Model, Input
from keras.layers import Dense, Dropout, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam
from keras.models import model_from_json, load_model
from keras.regularizers import l2, l1_l2

class DenseAutoEncoder():
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        pass

    def buildModel(self, inputDim, latentDim, lr=0.001):

        encoderIn = Input(batch_input_shape=(None, inputDim, 1))
        x = Flatten()(encoderIn)
        #x = Dense(1024, activation="selu", kernel_regularizer=l2(0.0001), kernel_initializer="he_normal")(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        #x = Dense(512, activation="selu", kernel_regularizer=l2(0.00001), kernel_initializer="he_normal")(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        x = Dense(256, activation="selu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        encoderOut = Dense(latentDim)(x)

        decoderIn = Input(shape=(latentDim,))
        x = Dense(128, activation="selu", kernel_initializer="he_normal")(decoderIn)
        x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        x = Dense(256, activation="selu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        #x = Dense(512, activation="selu", kernel_regularizer=l2(0.0001), kernel_initializer="he_normal")(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        #x = Dense(1024, activation="selu", kernel_regularizer=l2(0.0001), kernel_initializer="he_normal")(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.2)(x)
        x = Dense(inputDim)(x)
        decoderOut = Reshape((inputDim, 1))(x)

        self.encoder = Model(encoderIn, encoderOut)
        self.decoder = Model(decoderIn, decoderOut)
        self.autoencoder = Model( self.encoder.input, self.decoder(self.encoder(encoderIn)) )

        self.autoencoder.compile(optimizer=Adam(lr), loss="mse")
        self.encoder.summary()
        self.autoencoder.summary()
        return self

    def fit(self, x, batchSize, epochs):
        self.autoencoder.fit(x, x, batch_size=batchSize, epochs=epochs)
        return self

    def encode(self, x, batchSize=32, verbose=1):
        feats = self.encoder.predict(x, batchSize, verbose=verbose)
        return feats

    def saveEncoder(self, dir, name):
        print("saving feature extractor")
        featExtrJson = self.encoder.to_json()
        with open("{}encoder_{}_architecture.json".format(dir, name), "w") as json_file:
            json_file.write(featExtrJson)
        self.encoder.save_weights("{}encoder_{}_weights.h5".format(dir, name))
        pass

    def loadEncoder(self, dir, name):
        json_file = open("{}encoder_{}_architecture.json".format(dir, name), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        featExtr = model_from_json(loaded_model_json)

        # load weights into new model
        featExtr.load_weights("{}encoder_{}_weights.h5".format(dir, name))
        featExtr.compile(optimizer="adam", loss="mse")
        featExtr.summary()
        return featExtr
