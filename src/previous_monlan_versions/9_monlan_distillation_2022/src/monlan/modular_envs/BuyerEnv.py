import numpy as np

from src.monlan.modular_envs.EnvInterface import EnvInterface
from src.monlan.modular_envs.StateDescriptor import StateDescriptor

class BuyerEnv(EnvInterface):
    def __init__(self, feature_generator, spread_coef, lot_coef, lot_size, turn_off_spread):

        self.spread_coef = spread_coef
        self.lot_coef = lot_coef
        self.lot_size = lot_size
        self.turn_off_spread = turn_off_spread

        self.env_name = "buyer"

        class BuyerActionSpace():
            def __init__(self):
                self.actionsDict = { 0: "hold", 1: "buy" }
                self.n = 2
                pass
        self.action_space = BuyerActionSpace()

        self.feature_generator = feature_generator
        self.open_point = None

        self.takeprofit_puncts = None
        self.stoploss_puncts = None

        pass

    def get_name(self):
        return self.env_name

    def get_action_space(self):
        return self.action_space

    def get_feature_shape(self):
        return self.feature_generator.get_feature_shape()

    def reset(self):
        pass

    def set_open_point(self, history_step):
        self.open_point = history_step
        pass

    def set_limit_orders(self, takeprofit_puncts, stoploss_puncts):
        self.takeprofit_puncts = takeprofit_puncts
        self.stoploss_puncts = stoploss_puncts


    def step(self, history_price_array, price_feature_names_dict, history_step_id, action_id):

        if self.open_point > history_step_id:
            raise ValueError("Wrong order of open and close points in buyer env")

        reward_dict = {}
        m = history_step_id
        current_close_price = history_price_array[m][price_feature_names_dict["close"]]
        current_open_price = history_price_array[m][price_feature_names_dict["open"]]
        current_spread = history_price_array[m][price_feature_names_dict["spread"]]
        if self.turn_off_spread:
            current_hop_spread = 0.0

        i = history_step_id + 1
        one_hop_close_price = history_price_array[i][ price_feature_names_dict["close"] ]
        one_hop_open_price = history_price_array[i][ price_feature_names_dict["open"] ]
        one_hop_spread = history_price_array[i][ price_feature_names_dict["spread"] ]
        if self.turn_off_spread:
            one_hop_spread = 0.0

        j = history_step_id + 2
        two_hop_close_price = history_price_array[j][ price_feature_names_dict["close"] ]
        two_hop_open_price = history_price_array[j][ price_feature_names_dict["open"] ]
        two_hop_spread = history_price_array[j][ price_feature_names_dict["spread"] ]
        if self.turn_off_spread:
            two_hop_spread = 0.0

        k = self.open_point
        open_point_close_price = history_price_array[k][ price_feature_names_dict["close"] ]
        open_point_open_price = history_price_array[k][ price_feature_names_dict["open"] ]
        open_point_spread = history_price_array[k][ price_feature_names_dict["spread"] ]
        if self.turn_off_spread:
            open_point_spread = 0.0

        reward_dict[0] = (two_hop_open_price - (open_point_open_price + self.spread_coef * open_point_spread)) * self.lot_coef * self.lot_size
        reward_dict[1] = (one_hop_open_price - (open_point_open_price + self.spread_coef * open_point_spread)) * self.lot_coef * self.lot_size
        #reward_dict[0] = (two_hop_open_price - (current_open_price + self.spread_coef * current_spread)) * self.lot_coef * self.lot_size
        #reward_dict[1] = (one_hop_open_price - (current_open_price + self.spread_coef * current_spread)) * self.lot_coef * self.lot_size

        action = self.action_space.actionsDict[action_id]
        if action == "buy":
            if self.open_point is None:
                raise ValueError("Buyer open_point is None")
            current_obs = self.feature_generator.get_features(history_price_array, price_feature_names_dict, self.open_point)
            next_obs = self.feature_generator.get_features(history_price_array, price_feature_names_dict, history_step_id + 1)
        else:
            current_obs = self.feature_generator.get_features(history_price_array, price_feature_names_dict, history_step_id)
            next_obs = self.feature_generator.get_features(history_price_array, price_feature_names_dict, history_step_id + 1)

        info = {}

        ##############
        takeprofit_trigger = self.is_triggered_takeprofit(history_price_array, price_feature_names_dict, history_step_id)
        stoploss_trigger = self.is_triggered_stoploss(history_price_array, price_feature_names_dict, history_step_id)
        limit_trigger = self.check_limit_trigger( takeprofit_trigger, stoploss_trigger )
        if limit_trigger:
            reward_dict = self.get_limit_reward( takeprofit_trigger, stoploss_trigger )
            action_id = 1
            info["limit_triggered"] = True
        else:
            info["limit_triggered"] = False
        ##############

        done = False
        state_descriptor = StateDescriptor(current_obs, next_obs, action_id, reward_dict, done, info)

        return state_descriptor

    def is_triggered_stoploss(self, history_price_array, price_feature_names_dict, history_step_id):
        # stoploss
        info = {}
        info["stoploss"] = False

        k = self.open_point
        open_point_open_price = history_price_array[k][ price_feature_names_dict["open"] ]
        open_point_spread = history_price_array[k][ price_feature_names_dict["spread"] ]
        if self.turn_off_spread:
            open_point_spread = 0.0

        reward_dict = {}
        m = history_step_id
        current_low_price = history_price_array[m][price_feature_names_dict["low"]]
        if self.turn_off_spread:
            current_hop_spread = 0.0

        stoplossPrice = ((open_point_open_price - self.spread_coef * (self.stoploss_puncts - open_point_spread))) * self.lot_coef * self.lot_size
        next_extremum = current_low_price * self.lot_coef * self.lot_size
        if next_extremum - stoplossPrice <= 0:
            info["stoploss"] = True

        return info["stoploss"]

    def is_triggered_takeprofit(self, history_price_array, price_feature_names_dict, history_step_id):

        k = self.open_point
        open_point_open_price = history_price_array[k][ price_feature_names_dict["open"] ]
        open_point_spread = history_price_array[k][ price_feature_names_dict["spread"] ]
        if self.turn_off_spread:
            open_point_spread = 0.0

        reward_dict = {}
        m = history_step_id
        current_high_price = history_price_array[m][price_feature_names_dict["high"]]
        if self.turn_off_spread:
            current_hop_spread = 0.0

        # stoploss
        info = {}
        info["takeprofit"] = False
        takeprofit_price = ((open_point_open_price + self.spread_coef * (self.takeprofit_puncts + open_point_spread))) * self.lot_coef * self.lot_size
        next_extremum = current_high_price * self.lot_coef * self.lot_size
        if takeprofit_price - next_extremum <= 0:
            info["takeprofit"] = True

        return info["takeprofit"]

    def check_limit_trigger(self, tp_trigger, sl_trigger):

        limit_trigger = False
        if tp_trigger == True or sl_trigger == True:
            limit_trigger = True

        return limit_trigger

    def get_limit_reward(self, tp_trigger, sl_trigger):
        tp_reward = {}
        sl_reward = {}
        if tp_trigger:
            tp_reward[0] = self.spread_coef * self.takeprofit_puncts * self.lot_coef * self.lot_size
            tp_reward[1] = self.spread_coef * self.takeprofit_puncts * self.lot_coef * self.lot_size
        if sl_trigger:
            sl_reward[0] = -self.spread_coef * self.stoploss_puncts * self.lot_coef * self.lot_size
            sl_reward[1] = -self.spread_coef * self.stoploss_puncts * self.lot_coef * self.lot_size
        limit_reward = {}
        if tp_trigger == True and sl_trigger == True:
            # No one can say what was triggered first on history.
            # That's why choose negative scenario.
            limit_reward = sl_reward
            #limit_reward[0] = (abs(tp_reward[0]) + sl_reward[0]) / 2 # or neutral...
            #limit_reward[1] = (abs(tp_reward[1]) + sl_reward[1]) / 2 # or neutral...
        elif sl_trigger:
            limit_reward = sl_reward
        elif tp_trigger:
            limit_reward = tp_reward

        return limit_reward