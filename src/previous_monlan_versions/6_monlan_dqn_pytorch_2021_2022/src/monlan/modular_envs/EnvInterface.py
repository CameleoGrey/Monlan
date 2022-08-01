

from abc import ABCMeta, abstractmethod

class EnvInterface(metaclass=ABCMeta):

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def get_feature_shape(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, history_price_array, price_feature_names_dict, history_step_id, action_id):
        pass

    """@abstractmethod
    def is_done(self):
        pass"""

    """@abstractmethod
    def is_triggered_stoploss(self):
        pass

    @abstractmethod
    def is_triggered_takeprofit(self):
        pass

    @abstractmethod
    def get_limit_reward(self):
        pass"""