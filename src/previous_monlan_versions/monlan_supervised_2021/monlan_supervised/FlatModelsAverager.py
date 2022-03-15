
import numpy as np
from collections import deque

class FlatModelsAverager():
    def __init__(self, flattener, modelsPack):
        self.render = False
        self.load_model = False

        self.state_size = 1
        self.action_size = 1

        self.discount_factor = 1
        self.learning_rate = 1
        self.epsilon = 1
        self.epsilon_decay = 1
        self.epsilon_min = 1
        self.batch_size = 1
        self.train_start = 1
        self.memory = deque(maxlen=1)
        self.memorySize = 1
        self.fillMemoryByPretrainedModel = True
        ##########
        self.trainFreq = int(0.001 * self.memorySize)
        self.trainWaitStep = 0
        # self.rewardScaler = StandardScaler()
        # self.rewardScaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        ##########

        self.flattener = flattener
        self.models = modelsPack
        pass

    def preprocInput(self, x):
        x = self.flattener.flattenizeFeats(x, verbose=0)
        x = self.flattener.compressVecs(x, 1, verbose=0)
        return x

    def fit(self, x, y):
        pass

    def predict(self, x):

        x = self.preprocInput(x)

        predicts = []
        for model in self.models:
            y_pred = model.predict(x)[0]
            predicts.append(y_pred)
        y_avg = np.mean(predicts, axis=0)
        return y_avg

    def get_action(self, state):
        q_value = self.predict(state)

        return np.argmax(q_value[0])