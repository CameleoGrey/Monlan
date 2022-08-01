import torch.nn
import torchvision.models

import random
import numpy as np
from collections import deque
from datetime import datetime
from scipy.special import softmax
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class SupervisedOpener:
    def __init__(self, name, state_size, action_size,
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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def set_model(self, model, move_to_gpu=True):

        self.model = model
        if move_to_gpu:
            self.model.to( self.device )

    def get_name(self):
        return self.name

    def get_action(self, state):

        # 1d conv
        state = state.reshape((1, state.shape[1], state.shape[2]))

        # 2d conv
        #state = state.reshape((1, 1, state.shape[1], state.shape[2]))

        state = torch.Tensor( state ).to( self.device )

        ########################################
        # eps greed strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                start = datetime.now()
                q_value = self.model(state).cpu().detach().numpy()

                hold_q_value = 0.05

                """max_q = np.max(np.abs( q_value ))
                min_q = np.min(np.abs( q_value ))
                deal_potential = max_q / min_q
                open_threshold = 20.0
                base_hold_value = 0.00
                if deal_potential < open_threshold:
                    hold_q_value = 1.0
                else:
                    hold_q_value = base_hold_value"""


                """buy_threshold = 0.04
                sell_threshold = 0.08
                q_value[0][0] -= buy_threshold
                q_value[0][1] -= sell_threshold
                hold_q_value = 0.0"""

                """buy_threshold = -0.5
                sell_threshold = -0.5
                q_value[0][0] *= buy_threshold
                q_value[0][1] *= sell_threshold
                hold_q_value = 0.0"""

                q_value = np.array( [q_value[0][0], hold_q_value, q_value[0][1]] )
                total = datetime.now() - start
                chosen_action = np.argmax(q_value)
                return chosen_action
        #########################################

    def append_sample(self, state, action, reward, next_state, done):
        pass

    def train_model(self):
        pass

    def build_model(self):

        return None

    def build_flat_model(self):
        return None

    def build_conv_2d_model(self):

        return None

    def update_target_model(self):
        pass