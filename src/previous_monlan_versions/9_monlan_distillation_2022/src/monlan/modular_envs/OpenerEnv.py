
from src.monlan.modular_envs.EnvInterface import EnvInterface
from src.monlan.modular_envs.StateDescriptor import StateDescriptor


class OpenerEnv(EnvInterface):
    def __init__(self, feature_generator, spread_coef, lot_coef, lot_size, turn_off_spread):

        self.spread_coef = spread_coef
        self.lot_coef = lot_coef
        self.lot_size = lot_size
        self.turn_off_spread = turn_off_spread

        self.env_name = "opener"

        class OpenerActionSpace():
            def __init__(self):
                self.actionsDict = { 0: "buy", 1: "hold", 2: "sell" }
                self.n = 3
                pass
        self.action_space = OpenerActionSpace()

        self.feature_generator = feature_generator
        self.open_point = None
        self.close_point = None

        pass

    def get_name(self):
        return self.env_name

    def get_action_space(self):
        return self.action_space

    def get_feature_shape(self):
        return self.feature_generator.get_feature_shape()

    def reset(self):
        pass

    def step(self, history_price_array, price_feature_names_dict, history_step_id, action_id):

        reward_dict = {0: 0.0, 1: 0.0, 2: 0.0}
        action = self.action_space.actionsDict[action_id]

        # check end of the history

        if action == "hold":
            i = history_step_id + 1
            close_price = history_price_array[i][ price_feature_names_dict["close"] ]
            open_price = history_price_array[i][ price_feature_names_dict["open"] ]
            spread = history_price_array[i][ price_feature_names_dict["spread"] ]
            if self.turn_off_spread:
                spread = 0.0
            opener_hold_loss = 0.66 * (abs(close_price - open_price) + self.spread_coef * spread) * self.lot_coef * self.lot_size
            #opener_hold_loss = -0.05 * (abs(close_price - open_price) + self.spread_coef * spread) * self.lot_coef * self.lot_size
            reward_dict[action_id] = opener_hold_loss


        current_obs = self.feature_generator.get_features(history_price_array, price_feature_names_dict, history_step_id)
        next_obs = self.feature_generator.get_features(history_price_array, price_feature_names_dict, history_step_id + 1)

        info = {}
        info["deposite_delta"] = 0.0
        done = False
        state_descriptor = StateDescriptor(current_obs, next_obs, action_id, reward_dict, done, info)

        return state_descriptor


    def set_open_point(self, history_step):
        self.open_point = history_step
        pass

    def set_close_point(self, close_point):
        self.close_point = close_point
        pass

    def get_opener_close_deal_state(self, history_price_array, price_feature_names_dict, deal_type):

        if self.open_point is None:
            raise ValueError("Opener open_point is NaN")
        if self.close_point is None:
            raise ValueError("Opener close_point is NaN")
        if self.open_point > self.close_point:
            raise ValueError("Wrong order of open and close points in opener env")

        info = {}
        reward_dict = {0: 0.0, 1: 0.0, 2: 0.0}

        start_point_close_price = history_price_array[self.open_point][ price_feature_names_dict["close"] ]
        start_point_open_price = history_price_array[self.open_point][ price_feature_names_dict["open"] ]
        start_point_spread = history_price_array[self.open_point][ price_feature_names_dict["spread"] ]
        if self.turn_off_spread:
            start_point_spread = 0.0

        end_point_close_price = history_price_array[self.close_point][ price_feature_names_dict["close"] ]
        end_point_open_price = history_price_array[self.close_point][ price_feature_names_dict["open"] ]
        end_point_spread = history_price_array[self.close_point][ price_feature_names_dict["spread"] ]
        if self.turn_off_spread:
            end_point_spread = 0.0

        done = False
        start_obs = self.feature_generator.get_features(history_price_array, price_feature_names_dict, self.open_point)
        end_obs = self.feature_generator.get_features(history_price_array, price_feature_names_dict, self.close_point)

        if deal_type == "buy":
            buy_action_id = 0
            reward_dict[buy_action_id] = (end_point_open_price - (start_point_open_price + self.spread_coef * start_point_spread)) * self.lot_coef * self.lot_size
            info["deposit_delta"] = reward_dict[buy_action_id]
            state_descriptor = StateDescriptor( start_obs, end_obs, buy_action_id, reward_dict, done, info )
        elif deal_type == "sell":
            sell_action_id = 2
            reward_dict[sell_action_id] = (start_point_open_price - (end_point_open_price + self.spread_coef * end_point_spread)) * self.lot_coef * self.lot_size
            info["deposit_delta"] = reward_dict[sell_action_id]
            state_descriptor = StateDescriptor( start_obs, end_obs, sell_action_id, reward_dict, done, info )
        else:
            raise ValueError("Unknown deal_type")


        return state_descriptor

    def is_done(self):
        pass