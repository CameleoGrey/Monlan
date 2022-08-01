
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
            Reshape, GlobalAveragePooling2D, Layer)
import numpy as np
from keras.models import model_from_json, load_model
from keras.optimizers import Adam

class BVAE():
    def __init__(self):
        self.bvae = None
        pass

    def buildModel(self, inputShape, latentSize):
        encoder = Darknet19Encoder(inputShape, latentSize=latentSize, latentConstraints='bvae', beta=1)
        decoder = Darknet19Decoder(inputShape, latentSize=latentSize)
        self.bvae = AutoEncoder(encoder, decoder)
        self.bvae.ae.compile(optimizer=Adam(lr=0.001), loss='mse')
        return self

    def fit(self, x, epochs=100, batch_size=8):
        self.bvae.ae.fit(x, x, epochs=epochs, batch_size=batch_size)
        return self

    def encode(self, x, batchSize):
        encoded = self.bvae.encoder.predict(x, batch_size=batchSize)
        return encoded

    def decode(self, x):
        decoded = self.bvae.decoder.predict(x)
        return decoded

    def save(self, dir, name):
        modelJson = self.bvae.encoder.to_json()
        with open("{}{}_encoder_architecture.json".format(dir, name), "w") as json_file:
            json_file.write(modelJson)
        self.bvae.encoder.save_weights("{}{}_encoder_weights.h5".format(dir, name))

        modelJson = self.bvae.decoder.to_json()
        with open("{}{}_decoder_architecture.json".format(dir, name), "w") as json_file:
            json_file.write(modelJson)
        self.bvae.decoder.save_weights("{}{}_decoder_weights.h5".format(dir, name))
        pass

    def load(self, dir, name):
        json_file = open("{}{}_encoder_architecture.json".format(dir, name), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        encoder = model_from_json(loaded_model_json, custom_objects={'SampleLayer': SampleLayer()})
        encoder.load_weights("{}{}_encoder_weights.h5".format(dir, name))
        encoder.compile(optimizer="adam", loss="mse")
        encoder.summary()

        json_file = open("{}{}_decoder_architecture.json".format(dir, name), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        decoder = model_from_json(loaded_model_json, custom_objects={'SampleLayer': SampleLayer()})
        decoder.load_weights("{}{}_decoder_weights.h5".format(dir, name))
        decoder.compile(optimizer="adam", loss="mse")
        decoder.summary()

        return encoder, decoder

class AutoEncoder(object):
    def __init__(self, encoderArchitecture,
                 decoderArchitecture):
        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model
        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))
        print(self.ae.summary())


class SampleLayer(Layer):
    '''
    Keras Layer to grab a random sample from a distribution (by multiplication)
    Computes "(normal)*logvar + mean" for the vae sampling operation
    (written for tf backend)

    Additionally,
        Applies regularization to the latent space representation.
        Can perform standard regularization or B-VAE regularization.

    call:
        pass in mean then logvar layers to sample from the distribution
        ex.
            sample = SampleLayer('bvae', 16)([mean, logvar])
    '''

    def __init__(self, **kwargs):
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # save the shape for distribution sampling
        super(SampleLayer, self).build(input_shape)  # needed for layers

    def call(self, x, training=None):
        if len(x) != 2:
            raise Exception('input layers must be a list: mean and logvar')
        if len(x[0].shape) != 2 or len(x[1].shape) != 2:
            raise Exception('input shape is not a vector [batchSize, latentSize]')

        mean = x[0]
        logvar = x[1]

        # trick to allow setting batch at train/eval time
        if mean.shape[0] is None or logvar.shape[0] is None:
            return mean + 0 * logvar  # Keras needs the *0 so the gradinent is not None

        # kl divergence:
        latent_loss = -0.5 * (1 + logvar
                                  - K.square(mean)
                                  - K.exp(logvar))
        latent_loss = K.sum(latent_loss, axis=-1)  # sum over latent dimension
        latent_loss = K.mean(latent_loss, axis=0)  # avg over batch

        # use beta to force less usage of vector space:
        # set beta
        latent_loss = 1.0 * latent_loss
        self.add_loss(latent_loss)
        #self.add_loss(latent_loss, x)

        def reparameterization_trick():
            epsilon = K.random_normal(shape=logvar.shape,
                                      mean=0., stddev=1.)
            stddev = K.exp(logvar * 0.5)
            return mean + stddev * epsilon

        return K.in_train_phase(reparameterization_trick, mean + 0 * logvar,
                                training=training)  # TODO figure out why this is not working in the specified tf version???

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class ConvBnLRelu(object):
    def __init__(self, filters, kernelSize, strides=1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
    # return conv + bn + leaky_relu model
    def __call__(self, net, training=None):
        net = Conv2D(self.filters, self.kernelSize, strides=self.strides, padding='same', kernel_initializer="he_normal")(net)
        net = BatchNormalization()(net, training=training)
        net = LeakyReLU()(net)
        return net

class Architecture(object):
    '''
    generic architecture template
    '''
    def __init__(self, inputShape=None, batchSize=None, latentSize=None):
        '''
        params:
        ---------
        inputShape : tuple
            the shape of the input, expecting 3-dim images (h, w, 3)
        batchSize : int
            the number of samples in a batch
        latentSize : int
            the number of dimensions in the two output distribution vectors -
            mean and std-deviation
        latentSize : Bool or None
            True forces resampling, False forces no resampling, None chooses based on K.learning_phase()
        '''
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.latentSize = latentSize

        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')


class Darknet19Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Regularization may be applied to the latent space output

    a simple, fully convolutional architecture inspried by 
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, inputShape=(256, 256, 3), batchSize=None,
                 latentSize=1000, latentConstraints='bvae', beta=1., training=None):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.training=training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the netowrk
        inLayer = Input(self.inputShape, self.batchSize)
        net = ConvBnLRelu(32, kernelSize=3)(inLayer, training=self.training) # 1
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training) # 2
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training) # 3
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training) # 4
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training) # 5
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training) # 6
        net = ConvBnLRelu(128, kernelSize=1)(net, training=self.training) # 7
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training) # 8
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 9
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training) # 10
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 11
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training) # 12
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training) # 13
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 14
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training) # 15
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 16
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training) # 17
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training) # 18

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        logvar = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        logvar = GlobalAveragePooling2D()(logvar)

        sample = SampleLayer()([mean, logvar], training=self.training)

        return Model(inputs=inLayer, outputs=sample)

class Darknet19Decoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=None, latentSize=1000, training=None):
        self.training=training
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latentSize))(inLayer)
        # darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
        net = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(net)

        # TODO try inverting num filter arangement (e.g. 512, 1204, 512, 1024, 512)
        # and also try (1, 3, 1, 3, 1) for the filter shape
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(1024, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(512, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(128, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(256, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training)
        net = ConvBnLRelu(128, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net, training=self.training)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(32, kernelSize=3)(net, training=self.training)
        net = ConvBnLRelu(64, kernelSize=1)(net, training=self.training)

        # net = ConvBnLRelu(3, kernelSize=1)(net, training=self.training)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                      padding='same', activation="sigmoid")(net)

        return Model(inLayer, net)
