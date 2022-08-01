import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from src.monlan.modular_agents.ResnetBuilder import ResnetBuilder

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from scipy.special import softmax

class DQNAgent:
    def __init__(self, name, state_size, action_size,
                 discount_factor = 0.99,
                 learning_rate = 0.001,
                 epsilon = 1.0,
                 epsilon_decay = 0.9995,
                 epsilon_min = 0.01,
                 batch_size = 64,
                 train_start = 1500,
                 memorySize = 2000):

        self.name = name

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
        ##########
        self.trainFreq = int(0.001*self.memorySize)
        self.trainWaitStep = 0
        ##########

        self.model = self.build_model()

    def get_name(self):
        return self.name

    def get_action(self, state):
        if len(self.memory) <= self.train_start:
            # default
            #rand_action = random.randrange(self.action_size)

            # boost closer holds
            if self.action_size == 3:
                #rand_action = np.random.choice([0, 1, 2], p=[0.33, 0.34, 0.33])
                ################
                rand_action = np.random.choice([0, 1, 2], p=[0.45, 0.10, 0.45])
                ################
            elif self.action_size == 2:
                rand_action = np.random.choice([0, 1], p=[0.9, 0.1])
            else:
                raise ValueError("Broken action size")
            return rand_action

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        #########
        # mix of my strategies
        if self.epsilon == 0.0:
            #####################################################
            if self.action_size == 3:
                eps_action = np.random.choice([0, 1, 2], p=[0.45, 0.10, 0.45])
                return eps_action
            #####################################################
            q_value = np.array(self.model(state, training=False))[0]
            greed_action_id = np.argmax(q_value)
            return greed_action_id

        if self.action_size == 3:
            if np.random.rand() <= self.epsilon:
                #eps_action = np.random.choice([0, 1, 2], p=[0.33, 0.34, 0.33])
                #########################
                eps_action = np.random.choice([0, 1, 2], p=[0.45, 0.10, 0.45])
                #########################
                return eps_action
            else:
                q_value = np.array(self.model(state, training=False))[0]
                eps_greed_action = np.argmax(q_value)
                return eps_greed_action
        else:
            beta = 0.05
            q_value = np.array(self.model(state, training=False))[0]
            greed_action_id = np.argmax(q_value)
            probs = softmax(q_value)
            for i in range(len(probs)):
                if i == greed_action_id:
                    probs[i] = probs[i] - beta * ( 1.0 - probs[i] )
                else:
                    probs[i] = probs[i] - beta * ( 0.0 - probs[i] )
            pursuit_greed_action = np.argmax(probs)
            return pursuit_greed_action


        #########
        # my original pursuit version of exploration/exploitation
        """start = datetime.now()
        q_value = np.array(self.model(state, training=False))[0]
        total = datetime.now() - start
        greed_action_id = np.argmax(q_value)
        if self.epsilon == 0.0:
            return greed_action_id

        probs = softmax(q_value)
        if self.action_size == 3:
            hold_beta = 0.05
            deal_beta = 0.05
            for i in range(len(probs)):
                if i == 1:
                    probs[i] = probs[i] + hold_beta * ( 0.5 - probs[i] )
                elif i == greed_action_id:
                    probs[i] = probs[i] - deal_beta * ( 1.0 - probs[i] )
                else:
                    probs[i] = probs[i] - deal_beta * ( 0.0 - probs[i] )
        else:
            beta = 0.05
            for i in range(len(probs)):
                if i == greed_action_id:
                    probs[i] = probs[i] - beta * ( 1.0 - probs[i] )
                else:
                    probs[i] = probs[i] - beta * ( 0.0 - probs[i] )
        pursuit_greed_action = np.argmax(probs)
        return pursuit_greed_action"""

        #########
        # classic pursuit
        """q_value = np.array(self.model(state, training=False))[0]
        greed_action_id = np.argmax(q_value)
        if self.epsilon == 0.0:
            return greed_action_id
        
        beta = 0.001
        probs = softmax(q_value)
        for i in range(len(probs)):
            if i == greed_action_id:
                probs[i] = probs[i] + beta * ( 1.0 - probs[i] )
            else:
                probs[i] = probs[i] + beta * ( 0.0 - probs[i] )

        pursuit_greed_action = np.argmax(probs)
        return pursuit_greed_action"""
        ###########
        #epsilon greed policy
        """if np.random.rand() <= self.epsilon:

            # default
            #rand_action = random.randrange(self.action_size)

            # boost closer holds
            if self.action_size == 3:
                rand_action = np.random.choice([0, 1, 2], p=[0.33, 0.34, 0.33])
            elif self.action_size == 2:
                rand_action = np.random.choice([0, 1], p=[0.9, 0.1])
            else:
                raise ValueError("Broken action size")
            return rand_action
        else:
            #start = datetime.now()
            #q_value = self.model.predict(state)
            #total = datetime.now() - start

            #start = datetime.now()
            q_value = np.array(self.model(state, training=False))[0]
            #total = datetime.now() - start
            #print(total)

            chosen_action = np.argmax(q_value)
            return chosen_action"""
        ##########


    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        """if self.trainWaitStep < self.trainFreq:
            self.trainWaitStep += 1
            return"""
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

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        #start = datetime.now()
        #target = self.model.predict(update_input)
        #total = datetime.now() - start

        #start = datetime.now()
        target = np.array(self.model(update_input, training=False))
        #total = datetime.now() - start


        #start = datetime.now()
        #target_val = self.model.predict(update_target)
        #total = datetime.now() - start

        #start = datetime.now()
        target_val = np.array(self.model(update_target, training=False))
        #total = datetime.now() - start

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))

        #start = datetime.now()
        self.model.fit(update_input, target, batch_size=self.batch_size,epochs=1, verbose=0)
        #total = datetime.now() - start
        #total = datetime.now() - start

    def build_model(self, compile=True):

        model = None
        stateLen = len(self.state_size)
        if stateLen == 1:
            model = self.buildFlatModel(compile=compile)
        elif stateLen == 2:
            model = self.buildConv2DModel(compile=compile)

        return model

    def buildFlatModel(self, compile=True):
        model = Sequential()

        model.add(Dense(32, input_dim=self.state_size[0], activation='linear',kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Dense(64, activation='linear',kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LeakyReLU())

        model.add(Dense(128, activation='linear',kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LeakyReLU())

        model.add(Dense(64, activation='linear',kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LeakyReLU())

        model.add(Dense(32, activation='linear',kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Dense(self.action_size, activation='tanh',kernel_initializer='glorot_uniform'))

        model.summary()
        if compile:
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def buildConv2DModel(self, compile=True):

        model = ResnetBuilder.build_resnet_18((1, self.state_size[0], self.state_size[1]), self.action_size, outputActivation="tanh")
        #model = ResnetBuilder.build_resnet_34( (1, self.state_size[0], self.state_size[1]), self.action_size, outputActivation="tanh")
        #model = ResnetBuilder.build_resnet_50((1, self.state_size[0], self.state_size[1]), self.action_size, outputActivation="tanh")
        model.summary()
        if compile:
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def save_agent(self, path, name):

        self.model.save_weights( path + name + "_main_model"  + ".h5")
        tmp_1  = self.model
        self.model = None
        import joblib
        with open(path + "/" + name + ".pkl", mode="wb") as agentFile:
            joblib.dump(self, agentFile)
        self.model = tmp_1
        pass

    def load_agent(self, path, name, dropSupportModel = False, compile=True):
        import joblib
        with open(path + "/" + name + ".pkl", mode="rb") as agentFile:
            loadedAgent = joblib.load(agentFile)
        self.model = None
        loadedAgent.model = loadedAgent.build_model(compile=compile)
        loadedAgent.model.load_weights(path + name + "_main_model"  + ".h5")

        return loadedAgent

    def update_target_model(self):
        pass