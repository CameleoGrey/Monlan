import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense, Input, Conv2D, Conv3D, Conv1D, MaxPool2D, MaxPool1D,\
    concatenate, Flatten, Lambda, Reshape, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.backend import int_shape, expand_dims
from legacy.classes.monlan.agents.ResnetBuilder import ResnetBuilder
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models
from sklearn.preprocessing import StandardScaler, MinMaxScaler
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class DQNAgent:
    def __init__(self, state_size, action_size,
                 discount_factor = 0.99,
                 learning_rate = 0.001,
                 epsilon = 1.0,
                 epsilon_decay = 0.9999,
                 epsilon_min = 0.01,
                 batch_size = 64,
                 train_start = 1500,
                 memorySize = 2000,
                 fillMemoryByPretrainedModel = False):
        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start
        self.memory = deque(maxlen=memorySize)
        self.memorySize = memorySize
        self.fillMemoryByPretrainedModel = fillMemoryByPretrainedModel
        ##########
        self.trainFreq = int(0.001*self.memorySize)
        self.trainWaitStep = 0
        #self.rewardScaler = StandardScaler()
        #self.rewardScaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        ##########

        self.model = self.build_model()

    def build_model(self):

        model = None
        stateLen = len(self.state_size)
        if stateLen == 1:
            model = self.buildFlatModel()
        elif stateLen == 2:
            model = self.buildConv2DModel()
        elif stateLen == 3:
            model = self.buildHierarhicalConvModel()

        return model

    def buildFlatModel(self):
        """model = Sequential()
                model.add(Dense(32, input_dim=self.state_size[0], activation='tanh', kernel_initializer='random_uniform'))
                model.add(Dense(64, activation='tanh', kernel_initializer='random_uniform'))
                # model.add(Dropout(0.2))
                model.add(Dense(128, activation='tanh', kernel_initializer='random_uniform'))
                # model.add(Dropout(0.2))
                model.add(Dense(64, activation='tanh', kernel_initializer='random_uniform'))
                # model.add(Dropout(0.2))
                model.add(Dense(32, activation='tanh', kernel_initializer='random_uniform'))
                model.add(Dense(self.action_size, activation='tanh', kernel_initializer='random_uniform'))
                model.summary()
                model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))"""
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size[0], activation='relu',
                        kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu',
                        kernel_initializer='he_normal'))
        # model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu',
                        kernel_initializer='he_normal'))
        # model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu',
                        kernel_initializer='he_normal'))
        # model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_normal'))
        model.add(Dense(self.action_size, activation='tanh',
                        kernel_initializer='glorot_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def buildConv2DModel(self):

        #My favorite model but the slowest
        #model = ResnetBuilder.build_resnet_18((1, self.state_size[0], self.state_size[1]), self.action_size,outputActivation="tanh")
        #model = ResnetBuilder.build_resnet_34( (1, self.state_size[0], self.state_size[1]), self.action_size, outputActivation="tanh")
        #model = ResnetBuilder.build_resnet_50((1, self.state_size[0], self.state_size[1]), self.action_size, outputActivation="linear")

        ###################################
        #conv_base = EfficientNetB0(include_top=False, weights=None,
        #                           input_shape=(self.state_size[0], self.state_size[1], 1))
        #headModel = conv_base.output
        #model = Dense(self.action_size, activation="linear")(Flatten()(headModel))
        #model = models.Model(conv_base.input, model)
        ####################################

        #model.summary()
        #model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        #############
        base_model = ResnetBuilder.build_resnet_34((1, self.state_size[0], self.state_size[1]), 2, outputActivation="softmax")
        base_model.load_weights("../models/resnet_ha_no_cake_softmax" + ".h5")

        x = base_model.layers[-2].output
        x = Dense(256, activation='selu')(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='selu')(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='selu')(x)
        predictions = Dense(self.action_size, activation='tanh')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for i, layer in enumerate(model.layers):
            print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in model.layers[:153]:
            layer.trainable = False
        for layer in model.layers[153:]:
            layer.trainable = True

        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        #############

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
                activation="tanh")(headModelInput)
            #headModel = BatchNormalization()(headModel)
            #headModel = Dropout(0.05)(headModel)
            #headModel = MaxPool2D(pool_size=(2, 2), padding="same")(headModel)
            headModel = Conv2D( filters=64, kernel_size=4, strides=(2, 2), padding="same",
                activation="tanh")(headModel)
            #headModel = BatchNormalization()(headModel)
            #headModel = Dropout(0.2)(headModel)
            #headModel = MaxPool2D(pool_size=(2, 2), padding="same")(headModel)
            headModel = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same",
                activation="tanh")(headModel)
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
            activation="tanh")(mergedHead)
        #bodyModel = BatchNormalization()(bodyModel)
        #bodyModel = Dropout(0.05)(bodyModel)
        #bodyModel = MaxPool2D(pool_size=(2, 2), padding="same")(bodyModel)
        bodyModel = Conv3D(filters=64, kernel_size=4, strides=(1, 2, 2), padding="same",
            activation="tanh")(bodyModel)
        #bodyModel = BatchNormalization()(bodyModel)
        #bodyModel = Dropout(0.2)(bodyModel)
        #bodyModel = MaxPool2D(pool_size=(2, 2), padding="same")(bodyModel)
        bodyModel = Conv3D(filters=64, kernel_size=3, strides=(1, 1, 1), padding="same",
            activation="tanh")(bodyModel)
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
        bodyModel = Dense( self.action_size, activation="tanh" )(bodyModel)
        model = Model(inputs=mainInput, outputs=bodyModel)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def fit_agent(self, env, nEpisodes, plotScores, saveFreq=5):
        scores, episodes = [], []
        for e in range(nEpisodes):
            done = False
            score = 0
            state = env.reset()
            #state = np.reshape(state, [1, self.state_size])

            n_step = 0

            while not done:
                # get action for the current state and go one step in environment
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                #next_state = np.reshape(next_state, [1, self.state_size])
                # if an action make the episode end, then gives penalty of -100
                # reward = reward if not done or score == 499 else -100

                # save the sample <s, a, r, s'> to the replay memory
                self.append_sample(state, action, reward, next_state, done)
                # every time step do the training
                self.train_model()
                # agent.update_target_model()
                score += reward
                state = next_state

                n_step += 1
                if n_step % self.memorySize == 0:
                    n_step = 0
                    self.update_target_model()

                if done:
                    # every episode update the target model to be same with model
                    self.update_target_model()
                    print("{}: ".format(env.iStep) + str(env.deposit))

                    # every episode, plot the play time
                    if plotScores == True:
                        import matplotlib.pyplot as plt
                        scores.append(score)
                        episodes.append(e)
                        plt.close()
                        plt.plot(episodes, scores, 'b')
                        plt.savefig("./test_dqn.png")
                        print("episode:", e, "  score:", score, "  memory length:",
                            len(self.memory), "  epsilon:", self.epsilon)

            # save the model
            if e % saveFreq == 0:
                self.save_agent("./", "checkpoint")
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
        self.model.save_weights( path + name + "_main_model"  + ".h5")

        tmp_1  = self.model
        self.model = None
        import joblib
        with open(path + "/" + name + ".pkl", mode="wb") as agentFile:
            joblib.dump(self, agentFile)
        self.model = tmp_1

        pass

    def load_agent(self, path, name, dropSupportModel = False):
        import joblib
        loadedAgent = None
        with open(path + "/" + name + ".pkl", mode="rb") as agentFile:
            loadedAgent = joblib.load(agentFile)

        self.model = None
        loadedAgent.model = loadedAgent.build_model()

        loadedAgent.model.load_weights(path + name + "_main_model"  + ".h5")

        return loadedAgent
        #pass

    def loadPretrainedWeights(self, dir, baseName, agentType):
        self.model.load_weights(dir + baseName + "_" + agentType + ".h5")
        return self


    def update_target_model(self):
        #self.target_model.set_weights(self.model.get_weights())
        pass

    def get_action(self, state):
        if len(self.memory) <= self.train_start and not self.fillMemoryByPretrainedModel:
            return random.randrange(self.action_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            #if len(q_value[0]) == 3:
            #    tmp = q_value[0][2]
            #    q_value[0][2] = q_value[0][1]
            #    q_value[0][1] = tmp

            #####
            #if self.action_size == 3:
            #    maxInd = np.argmax(q_value[0])
            #    q_value = list(q_value[0])
            #    q_value[maxInd] = -1000000000
            #    q_value = np.array(q_value)
            #    return np.argmax(q_value)
            #####
            #if len(q_value[0]) == 3:
            #    q_value[0][1] = -1e10

            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) <= self.train_start:
            return
        if self.trainWaitStep < self.trainFreq:
            self.trainWaitStep += 1
            return
        else:
            #print(self.trainWaitStep)
            """self.trainWaitStep = 0
            fitSampleSize = int(0.2 * len(self.memory))
            fitSample = random.sample(self.memory, fitSampleSize)
            fitSample = np.array(fitSample)[:, 2]
            rewardList = fitSample.reshape((-1, 1))
            self.rewardScaler.fit(rewardList)"""
            pass

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        dataShape = []
        dataShape.append(batch_size)
        for i in range(len(self.state_size)):
            dataShape.append(self.state_size[i])
        if len(mini_batch[0][0].shape) != 2:
            dataShape.append(1)
        dataShape = tuple(dataShape)

        update_input = np.zeros(dataShape)
        update_target = np.zeros(dataShape)
        action, reward, done = [], [], []

        """if len(mini_batch[0][0].shape) == 2:
            for i in range(self.batch_size):
                update_input[i] = np.reshape(mini_batch[i][0][0], (-1, 1))
                action.append(mini_batch[i][1])
                reward.append(mini_batch[i][2])
                update_target[i] = np.reshape(mini_batch[i][3][0], (-1, 1))
                done.append(mini_batch[i][4])
        else:"""
        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        ########################################
        #scale reward batch
        #reward = np.array(reward).reshape((-1, 1))
        #reward = self.rewardScaler.transform(reward)
        #reward = list(reward.reshape((-1,)))
        ########################################

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))

        ##################
        #for i in range(len(target)):
        #    tmp = target[i].reshape((-1, 1))
        #    tmp = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(tmp)
        #    tmp = tmp.reshape((-1))
        #    target[i] = tmp
        ##################

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)