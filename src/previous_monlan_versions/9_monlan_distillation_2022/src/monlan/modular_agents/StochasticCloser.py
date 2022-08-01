

import random
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class StochasticCloser:
    def __init__(self, name=None, state_size=None, action_size=2,
                 epsilon=0.0,
                 epsilon_decay=0.0,
                 epsilon_min=0.0):

        self.name = name
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min


    def get_name(self):
        return self.name

    def get_action_head_only(self, hold_proba, close_proba):

        if self.action_size == 2:
            action = np.random.choice([0, 1], p=[hold_proba, close_proba])
        else:
            raise ValueError("Closer must have action_size == 2")

        return action

    def get_action(self, hold_proba, close_proba):

        if self.action_size == 2:
            action = np.random.choice([0, 1], p=[hold_proba, close_proba])
        else:
            raise ValueError("Closer must have action_size == 2")

        return action

    def append_sample(self, state, action, reward, next_state, done):
        pass

    def train_model(self):
       pass

    def build_model(self):
        pass

    def build_conv_2d_model(self):
        pass

    def update_target_model(self):
        pass