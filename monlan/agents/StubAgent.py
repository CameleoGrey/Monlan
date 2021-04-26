import random
import numpy as np
from collections import deque
from keras.layers import Dense, Input, Conv2D, Conv3D, Conv1D, MaxPool2D, MaxPool1D,\
    concatenate, Flatten, Lambda, CuDNNLSTM, Reshape, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.backend import int_shape, expand_dims
from monlan.agents.ResnetBuilder import *

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class StubAgent:
    def __init__(self, state_size, action_size,
                 discount_factor = 0.99,
                 learning_rate = 0.001,
                 epsilon = 1.0,
                 epsilon_decay = 0.9999,
                 epsilon_min = 0.01,
                 batch_size = 64,
                 train_start = 1500,
                 memorySize = 2000):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start
        # create replay memory using deque
        self.memory = deque(maxlen=memorySize)
        self.memorySize = memorySize

        # create main model and target model
        self.model = self.build_model()
        #self.target_model = self.build_model()

        # initialize target model
        #self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        model = None
        stateLen = len(self.state_size)
        if stateLen == 1:
            model = self.buildFlatModel()
        elif stateLen == 2:
            model = self.buildSimpleConvModel()
        elif stateLen == 3:
            model = self.buildHierarhicalConvModel()

        return model

    def buildFlatModel(self):
        model = Sequential()
        model.add(Dense(1, input_dim=self.state_size[0], activation='relu',
            kernel_initializer='glorot_uniform'))
        #model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def buildSimpleConvModel(self):
        mainInputShape = (self.state_size[0], self.state_size[1], 1)
        mainInput = Input(shape=mainInputShape)
        headModel = Conv2D(filters=1, kernel_size=2, strides=(1, 1), padding="same", activation="relu")(mainInput)
        headModel = Flatten()(headModel)
        bodyModel = Dense(self.action_size, activation="tanh")(headModel)
        model = Model(inputs=mainInput, outputs=bodyModel)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def buildHierarhicalConvModel(self):

        headModels = []
        mainInputShape = ( self.state_size[0], self.state_size[1], self.state_size[2], 1)
        mainInput = Input(shape=mainInputShape)
        #split = Lambda(lambda x: tf.split(x, num_or_size_splits=self.state_size[0], axis=1))(mainInput)
        #y = Lambda(lambda x: x[:, 0, :, :], output_shape=(1,) + mainInputShape[2:])(mainInput)
        for i in range( self.state_size[0] ):
            #y = Lambda(lambda x: x[:, 0, :, :], output_shape=(1,) + mainInputShape[2:])(mainInput)
            #headModelInput = Input( shape=(self.state_size[1], self.state_size[2], 1) )(split[i])
            #headModel = Conv2D(filters=64, kernel_size=4, padding="same", activation="relu")(split[i])
            headModelInput = Lambda(lambda x: x[:, i, :, :], mainInputShape[1:])(mainInput)
            #headModelInput = Input(shape=(self.state_size[1], self.state_size[2], 1))(headModelInput)
            #headModelInput._keras_shape = (None, headModelInput.keras_shape[2], headModelInput.keras_shape[3], headModelInput.keras_shape[4] )
            headModel = Conv2D(filters=32, kernel_size=8, strides=(4, 4), padding="same",
                activation="elu")(headModelInput)
            #headModel = BatchNormalization()(headModel)
            #headModel = Dropout(0.05)(headModel)
            #headModel = MaxPool2D(pool_size=(2, 2), padding="same")(headModel)
            headModel = Conv2D( filters=64, kernel_size=4, strides=(2, 2), padding="same",
                activation="elu")(headModel)
            #headModel = BatchNormalization()(headModel)
            #headModel = Dropout(0.2)(headModel)
            #headModel = MaxPool2D(pool_size=(2, 2), padding="same")(headModel)
            headModel = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same",
                activation="elu")(headModel)
            #headModel = BatchNormalization()(headModel)
            #headModel = Dropout(0.2)(headModel)
            # headModel = MaxPool2D(pool_size=(2, 2), padding="same")(headModel)
            ###############################
            """headModel = Reshape((headModel._keras_shape[1] * headModel._keras_shape[2], headModel._keras_shape[3]))(headModel)
            headModel._keras_shape = (None, headModel._keras_shape[1], headModel._keras_shape[2])
            # bodyModel = Activation(activation="linear")(bodyModel)
            headModel = CuDNNLSTM(128)(headModel)
            headModel = Reshape((1, headModel._keras_shape[1]))(headModel)"""
            ###############################
            #headModel = Model(inputs=headModelInput, outputs=headModel)
            headModel = Reshape((1,int_shape(headModel)[1],
                                  int_shape(headModel)[2],
                                  int_shape(headModel)[3]))(headModel)
            #print(headModel.shape)
            headModels.append(headModel)

        #try to change axis of concatenation
        mergedHead = concatenate( [head for head in headModels], axis=1 )
        #print(mergedHead.shape)
        #######
        #mergedHead = Reshape((1, mergedHead._keras_shape[1], mergedHead._keras_shape[2]))(mergedHead)
        #######
        #mergedHead = expand_dims(mergedHead, axis=-1)
        featLen = int_shape(mergedHead)[1]
        rowLen = int_shape(mergedHead)[2] * int_shape(mergedHead)[3]
        w2vLen = int_shape(mergedHead)[4]
        #filterLen = int_shape(mergedHead)[3]
        #mergedHead._keras_shape = (None, mergedHead.shape[1].value, int_shape(mergedHead)[2], int_shape(mergedHead)[3])
        mergedHead = Reshape((featLen,
                              rowLen,
                             w2vLen, 1))(mergedHead)
        #######

        #try conv3d at top of head features
        # try conv1d at low levels
        #print(mergedHead.shape)
        featTypes = self.state_size[0]
        bodyModel = Conv3D(filters=32, kernel_size=8, strides=(1, 4, 4), padding="same",
            activation="elu")(mergedHead)
        #bodyModel = BatchNormalization()(bodyModel)
        #bodyModel = Dropout(0.05)(bodyModel)
        #bodyModel = MaxPool2D(pool_size=(2, 2), padding="same")(bodyModel)
        bodyModel = Conv3D(filters=64, kernel_size=4, strides=(1, 2, 2), padding="same",
            activation="elu")(bodyModel)
        #bodyModel = BatchNormalization()(bodyModel)
        #bodyModel = Dropout(0.2)(bodyModel)
        #bodyModel = MaxPool2D(pool_size=(2, 2), padding="same")(bodyModel)
        bodyModel = Conv3D(filters=64, kernel_size=3, strides=(1, 1, 1), padding="same",
            activation="elu")(bodyModel)
        #bodyModel = BatchNormalization()(bodyModel)
        #bodyModel = Dropout(0.2)(bodyModel)
        #bodyModel = MaxPool2D(pool_size=(2, 2), padding="same")(bodyModel)

        #####################################################
        #bodyModel = Reshape((bodyModel.shape[1] * bodyModel.shape[2], bodyModel.shape[3] * bodyModel.shape[4]))(bodyModel)
        #bodyModel._keras_shape = (None, bodyModel._keras_shape[1].value, bodyModel._keras_shape[2].value)
        #bodyModel = CuDNNLSTM(512, return_sequences=False)(bodyModel)
        ####################################################
        #bodyModel = CuDNNLSTM( 512, return_sequences=True )(bodyModel)
        #bodyModel = SeqSelfAttention()(bodyModel)
        ####################################################

        bodyModel = Flatten()(bodyModel)
        bodyModel = Dense( self.action_size, activation="linear" )(bodyModel)
        model = Model(inputs=mainInput, outputs=bodyModel)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def fit_agent(self, env, nEpisodes, plotScores, saveFreq=5):
        pass

    def use_agent(self, env):
        self.epsilon = 0
        done = False
        score = 0
        state = env.reset()
        #state = np.reshape(state, [1, self.state_size])
        while not done:
            # get action for the current state and go one step in environment
            action = self.get_action(state)
            next_state, reward, done, info = env.step(action)
            #next_state = np.reshape(next_state, [1, self.state_size])
            score += reward
            state = next_state
            if done:
                # every episode update the target model to be same with model
                print("{}: ".format(env.iStep) + str(env.deposit))
                print("score:", score)
        pass

    def save_agent(self, path, name):
        #import joblib
        #with open(path + "/" + name + ".pkl", mode="wb") as agentFile:
        #    joblib.dump(self, agentFile)

        #self.model.save( path + name + "main_model" )
        #self.target_model.save(path + name + "sup_model")
        self.model.save_weights( path + name + "_main_model"  + ".h5")
        #self.target_model.save_weights(path + name + "_sup_model" + ".h5")

        tmp_1  = self.model
        #tmp_2 = self.target_model
        self.model = None
        #self.target_model = None
        import joblib
        with open(path + "/" + name + ".pkl", mode="wb") as agentFile:
            joblib.dump(self, agentFile)
        self.model = tmp_1
        #self.target_model = tmp_2

        pass

    def load_agent(self, path, name, dropSupportModel = False):
        import joblib
        loadedAgent = None
        with open(path + "/" + name + ".pkl", mode="rb") as agentFile:
            loadedAgent = joblib.load(agentFile)

        self.model = None
        #self.target_model = None
        loadedAgent.model = loadedAgent.build_model()
        #if dropSupportModel == False:
            #loadedAgent.target_model = loadedAgent.build_model()
            #loadedAgent.update_target_model()

        loadedAgent.model.load_weights(path + name + "_main_model"  + ".h5")
        #if dropSupportModel == False:
        #    loadedAgent.target_model.load_weights(path + name + "_sup_model" + ".h5")

        return loadedAgent
        #pass

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        #self.target_model.set_weights(self.model.get_weights())
        pass

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        return 0

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        pass

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        pass