import torch.nn
import torchvision.models

from src.monlan.modular_agents.ResnetBuilder_pytorch import *
import random
import numpy as np
from collections import deque
from scipy.special import softmax
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

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

        ##########
        self.hold_count = 0
        ##########

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.build_model().to( self.device )
        #self.loss_function = torch.nn.MSELoss()
        self.loss_function = torch.nn.SmoothL1Loss()
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

    def get_name(self):
        return self.name

    def get_action(self, state):

        state = state.reshape((1, 1, state.shape[1], state.shape[2]))
        state = torch.Tensor( state ).to( self.device )

        #########
        # mix of my strategies
        if self.epsilon == 0.0:
            #####################################################
            if self.action_size == 3:
                eps_action = np.random.choice([0, 1, 2], p=[0.45, 0.10, 0.45])
                return eps_action
            #####################################################
            q_value = self.model(state).cpu().detach().numpy()
            greed_action_id = np.argmax(q_value)
            return greed_action_id

        from datetime import datetime
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
                rand_action = np.random.choice([0, 1], p=[0.8, 0.2])
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
            if np.random.rand() <= self.epsilon:
                #eps_action = np.random.choice([0, 1, 2], p=[0.33, 0.34, 0.33])
                #########################
                eps_action = np.random.choice([0, 1, 2], p=[0.45, 0.10, 0.45])
                #########################
                return eps_action
            else:
                q_value = self.model(state).cpu().detach().numpy()
                eps_greed_action = np.argmax(q_value)

                return eps_greed_action
        else:
            beta = 0.05
            q_value = self.model(state).cpu().detach().numpy()
            greed_action_id = np.argmax(q_value)
            probs = softmax(q_value)
            for i in range(len(probs)):
                if i == greed_action_id:
                    probs[i] = probs[i] + beta * ( 1.0 - probs[i] )
                else:
                    probs[i] = probs[i] + beta * ( 0.0 - probs[i] )
            pursuit_greed_action = np.argmax(probs)


            ##############################
            # trying to prevent hold overfit
            if pursuit_greed_action == 0:
                self.hold_count += 1
            if self.hold_count == 10:
                self.hold_count = 0
                pursuit_greed_action = 1
            ##############################

            return pursuit_greed_action


        #########
        # my original pursuit version of exploration/exploitation
        """start = datetime.now()
        q_value = self.model(state).cpu().detach().numpy()
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
            pursuit_greed_action = np.argmax(probs)

            ##############################
            # trying to prevent hold overfit
            if pursuit_greed_action == 1:
                self.hold_count += 1
            if self.hold_count == 10:
                self.hold_count = 0
                pursuit_greed_action = np.random.choice([0, 1, 2], p=[0.5, 0.0, 0.5])
            ##############################

            return pursuit_greed_action
        else:
            beta = 0.05
            for i in range(len(probs)):
                if i == greed_action_id:
                    probs[i] = probs[i] - beta * ( 1.0 - probs[i] )
                else:
                    probs[i] = probs[i] - beta * ( 0.0 - probs[i] )
            pursuit_greed_action = np.argmax(probs)

            ##############################
            # trying to prevent hold overfit
            if pursuit_greed_action == 0:
                self.hold_count += 1
            if self.hold_count == 20:
                self.hold_count = 0
                pursuit_greed_action = 1
            ##############################

            return pursuit_greed_action"""

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            return

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

        image_shape = mini_batch[0][0].shape

        update_input = update_input.reshape((self.batch_size, 1, image_shape[1], image_shape[2]))
        update_input = torch.Tensor(update_input).to( self.device )

        update_target = update_target.reshape((self.batch_size, 1, image_shape[1], image_shape[2]))
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

        model = None
        stateLen = len(self.state_size)
        if stateLen == 1:
            model = self.buildFlatModel()
        elif stateLen == 2:
            model = self.buildConv2DModel()

        return model

    def buildFlatModel(self):
        pass

    def buildConv2DModel(self):

        model = torchvision.models.resnet18(pretrained=False, progress=True)
        #model = torchvision.models.resnet34(pretrained=False, progress=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=self.action_size, bias=True)

        model = nn.Sequential(
            model,
            nn.Tanh()
        )
        #model = resnet18( self.action_size, activation="tanh" )
        #model = resnet34( self.action_size, activation="tanh" )

        return model

    def update_target_model(self):
        pass