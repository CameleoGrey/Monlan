
import numpy as np
from collections import deque
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class AlwaysHoldAgent:
    def __init__(self, name, state_size, action_size=2,
                 discount_factor = 0.99,
                 learning_rate = 0.001,
                 epsilon = 1.0,
                 epsilon_decay = 0.9995,
                 epsilon_min = 0.01,
                 batch_size = 64,
                 train_start = 1500,
                 memory_size = 2000):

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
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        ##########
        self.trainFreq = int(0.001*self.memory_size)
        self.trainWaitStep = 0
        ##########

        ##########
        self.hold_count = 0
        ##########

    def get_name(self):
        return self.name

    def get_action(self, state):

        return 0

    def append_sample(self, state, action, reward, next_state, done):
        pass

    def train_model(self):
        pass

    def build_model(self):

        pass

    def build_flat_model(self):
        pass

    def build_conv_2d_model(self):

        pass

    def update_target_model(self):
        pass