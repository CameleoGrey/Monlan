import random
import numpy as np
from collections import deque
from keras.layers import Dense, Input, Conv2D, Conv3D, Conv1D, MaxPool2D, MaxPool1D,\
    concatenate, Flatten, Lambda, Reshape, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.backend import int_shape, expand_dims
from monlan.agents.ResnetBuilder import *

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class AlwaysCloseAgent:
    def __init__(self, state_size, action_size, agentName):
        # if you want to see Cartpole learning, then change to True

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.name = agentName
        self.batch_size = 200
        self.epsilon = 0
        self.memory = []

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        pass

    def fit_agent(self, env, nEpisodes, plotScores, saveFreq=5):

        pass

    def use_agent(self, env):
        pass

    def save_agent(self, path, name):

        pass

    def load_agent(self, path, name, dropSupportModel = False):

        return self

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        #self.target_model.set_weights(self.model.get_weights())
        pass

    def loadPretrainedWeights(self, dir, baseName, agentType):
        return self

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if self.name == "opener":
            action = np.random.randint(low=0, high=2, size=1)
        else:
            action = 1

        return action

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        pass

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        pass