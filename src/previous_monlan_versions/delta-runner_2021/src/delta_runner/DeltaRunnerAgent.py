
import numpy as np

class DeltaRunnerAgent():
    def __init__(self, name, state_size, action_size, model, scaler, base_threshold):

        self.model = model
        self.scaler = scaler
        self.base_threshold = base_threshold

        self.name = name

        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.0

    def get_name(self):
        return self.name

    def get_action(self, state):
        normalized_state = state[0]
        normalized_state = self.scaler.transform( normalized_state ).reshape((1, -1))
        #normalized_state = self.scaler.transform(normalized_state).T

        y_pred = self.model.predict( normalized_state )
        y_pred = y_pred.reshape((-1, 1))
        y_pred = self.scaler.inverse_transform(y_pred)
        opener_threshold = 1.5 * self.base_threshold
        closer_threshold = 0.16 * self.base_threshold

        if self.name == "opener":
            if np.abs(y_pred) > opener_threshold:
                if np.sign( y_pred ) > 0:
                    return 0
                else:
                    return 2
            else:
                return 1
        else:
            return 1

        """elif self.name == "buyer":
            if y_pred > closer_threshold:
                return 0
            else:
                return 1

        elif self.name == "seller":
            if y_pred < -closer_threshold:
                return 0
            else:
                return 1
        else:
            raise Exception("Invalid agent name.")"""

    def append_sample(self, state, action, reward, next_state, done):
        pass

    def train_model(self):
        pass

    def build_model(self):

        return None

    def buildFlatModel(self):
        pass

    def buildConv2DModel(self):

        return None

    def update_target_model(self):
        pass

