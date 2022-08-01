
import torch
from torch import nn
from torchvision import models

from src.monlan.modular_agents.tcn.tcn import *


import random
import numpy as np
from collections import deque
from scipy.special import softmax
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#torch.set_warn_always(False)

class DQNAgent:
    def __init__(self, name, state_size, action_size,
                 discount_factor = 0.99,
                 learning_rate = 0.001,
                 epsilon = 1.0,
                 epsilon_decay = 0.9995,
                 epsilon_min = 0.01,
                 batch_size = 64,
                 train_start = 1500,
                 memory_size = 2000,
                 reward_comparison_steps = 40,
                 reward_scaler = None):

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

        ##############
        self.reward_comparison_steps = reward_comparison_steps
        self.ema_reward_alpha = 1.0 / float( self.reward_comparison_steps )
        self.ema_reward = 100.0 #initial value
        self.reward_scaler = reward_scaler
        ##############

        ##########
        self.hold_count = 0
        ##########

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.build_model().to( self.device )
        self.loss_function = torch.nn.MSELoss()
        #self.loss_function = torch.nn.SmoothL1Loss()

        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        #self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.learning_rate, betas=(0.66, 0.999), weight_decay=1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2, betas=(0.9, 0.999), amsgrad=True)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.0, weight_decay=5e-4)

    def get_name(self):
        return self.name

    def get_action_head_only(self, state):

        if isinstance( state, np.ndarray ):
            state = torch.from_numpy( state )
            state = state.to( self.device )

        with torch.no_grad():
            q_value = self.model.linear(state).cpu().detach().numpy()
        greed_action_id = np.argmax(q_value)

        return greed_action_id

    def get_action(self, state):

        # 1d conv
        state = state.reshape(( 1, state.shape[1], state.shape[2]))

        # 2d conv
        #state = state.reshape((1, 1, state.shape[1], state.shape[2]))
        
        state = torch.Tensor( state ).to( self.device )

        #########
        # mix of my strategies
        if self.epsilon == 0.0:
            #####################################################
            # action size == 3 means opener. In this version is assumed that
            # in composite agent will be used supervised opener that has
            # his own policy.
            if self.action_size == 3:
                eps_action = np.random.choice([0, 1, 2], p=[0.45, 0.10, 0.45])
                return eps_action
            #####################################################


            ####################################################
            # greed
            q_value = self.model(state).cpu().detach().numpy()
            greed_action_id = np.argmax(q_value)
            ####################################################

            return greed_action_id

        if len(self.memory) <= self.train_start:
            # default
            #rand_action = random.randrange(self.action_size)

            # boost closer holds
            if self.action_size == 3:
                #rand_action = np.random.choice([0, 1, 2], p=[0.33, 0.34, 0.33])
                ################
                rand_action = np.random.choice([0, 1, 2], p=[0.33, 0.34, 0.33])
                ################
            elif self.action_size == 2:
                rand_action = np.random.choice([0, 1], p=[0.95, 0.05])
                #rand_action = np.random.choice([0, 1], p=[0.5, 0.5])
            else:
                raise ValueError("Broken action size")

            return rand_action

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        ########################################
        # eps greed strategy
        """if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                start = datetime.now()
                q_value = self.model(state).cpu().detach().numpy()
                total = datetime.now() - start
                chosen_action = np.argmax(q_value[0])
                return chosen_action"""
        #########################################

        if self.action_size == 3:

            ####################################################################
            # pursuit greed
            """beta = 0.001
            q_value = self.model(state).cpu().detach().numpy()
            greed_action_id = np.argmax(q_value)
            probs = softmax(q_value)
            for i in range(len(probs)):
                if i == greed_action_id:
                    probs[i] = probs[i] + beta * ( 1.0 - probs[i] )
                else:
                    probs[i] = probs[i] + beta * ( 0.0 - probs[i] )
            pursuit_greed_action = np.argmax(probs)
            return pursuit_greed_action"""

            ###################################################################

            ###################################################################
            # eps greed
            if np.random.rand() <= self.epsilon:
                eps_action = np.random.choice([0, 1, 2], p=[0.33, 0.34, 0.33])
                #########################
                #eps_action = np.random.choice([0, 1, 2], p=[0.45, 0.10, 0.45])
                #########################
                return eps_action
            else:
                q_value = self.model(state).cpu().detach().numpy()
                eps_greed_action = np.argmax(q_value)

                return eps_greed_action
            ####################################################################

        else:
            # pursuit greed
            """beta = 0.001
            q_value = self.model(state).cpu().detach().numpy()
            greed_action_id = np.argmax(q_value)
            probs = softmax(q_value)
            for i in range(len(probs)):
                if i == greed_action_id:
                    probs[i] = probs[i] + beta * ( 1.0 - probs[i] )
                else:
                    probs[i] = probs[i] + beta * ( 0.0 - probs[i] )
            pursuit_greed_action = np.argmax(probs)"""

            # eps greed
            if np.random.rand() <= self.epsilon:
                pursuit_greed_action = np.random.choice([0, 1], p=[0.95, 0.05])
                # pursuit_greed_action = np.random.choice([0, 1], p=[0.5, 0.5])
            else:
                q_value = self.model(state).cpu().detach().numpy()
                pursuit_greed_action = np.argmax(q_value)


            ##############################
            # trying to prevent hold overfit
            """if pursuit_greed_action == 0:
                self.hold_count += 1
            if pursuit_greed_action == 1:
                self.hold_count = 0
            if self.hold_count >= 60:
                self.hold_count = 0
                pursuit_greed_action = 1"""
            ##############################

            return pursuit_greed_action

    def append_sample(self, state, action, reward, next_state, done):

        # compare to previous N deals
        raw_reward = reward
        reward = reward / (abs(self.ema_reward) + 1.0)
        #reward = reward - np.sign(self.ema_reward)
        if len(self.memory) >= self.reward_comparison_steps:
            if self.action_size == 2 and action == 1:
                self.ema_reward = self.ema_reward_alpha * raw_reward + (1.0 - self.ema_reward_alpha) * ( self.ema_reward )

        if self.reward_scaler is not None:
            reward = self.reward_scaler.transform( reward )

        # Idea: don't push outliers into memory
        #if np.abs( reward ) > 1.0:
        #    return

        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        data_shape = []
        data_shape.append(batch_size)
        for i in range(len(self.state_size)):
            data_shape.append(self.state_size[i])
            
        # 2d conv
        #if len(mini_batch[0][0].shape) != 2:
        #    dataShape.append(1)
        #dataShape = tuple(dataShape)

        update_input = np.zeros(data_shape)
        update_target = np.zeros(data_shape)
        action, reward, done = [], [], []

        for i in range(self.batch_size):
        
            # 1d conv
            update_input[i] = mini_batch[i][0].reshape((mini_batch[i][0].shape[0], mini_batch[i][0].shape[1], mini_batch[i][0].shape[2]))
            update_target[i] = mini_batch[i][3].reshape((mini_batch[i][3].shape[0], mini_batch[i][3].shape[1], mini_batch[i][3].shape[2]))
            
            #update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            #update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        image_shape = mini_batch[0][0].shape

        # 2d conv
        #update_input = update_input.reshape((self.batch_size, 1, image_shape[1], image_shape[2]))
        update_input = torch.Tensor(update_input).to( self.device )

        #2d conv
        #update_target = update_target.reshape((self.batch_size, 1, image_shape[1], image_shape[2]))
        update_target = torch.Tensor(update_target).to( self.device )

        target = self.model(update_input).cpu().detach().numpy()
        target_val = self.model(update_target).cpu().detach().numpy()

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))

        target = torch.Tensor(target).to( self.device )

        pred = self.model(update_input)
        loss = self.loss_function(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def build_model(self):

        model = self.build_conv_2d_model()

        return model

    def build_conv_2d_model(self):

        #model = torchvision.models.resnet18(pretrained=False, progress=True)
        #model = torchvision.models.resnet34(pretrained=False, progress=True)
        #model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #model.fc = nn.Linear(in_features=512, out_features=self.action_size, bias=True)

        model = TCNDenseOutput( input_size=7, output_size=self.action_size, num_channels=[64]*16, kernel_size=8, dropout=0.33 )

        return model

    def update_target_model(self):
        pass