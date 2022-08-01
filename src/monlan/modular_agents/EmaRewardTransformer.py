
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

class EmaRewardTransformer():
    def __init__(self):

        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))

        pass

    def fit(self, history_df, spread_coef, lot_coef, lot_size, deal_length=60, averaging_n=40, history_step_size=10):

        alpha = 1.0 / averaging_n
        buy_abs_rewards, sell_abs_rewards = self.collect_rewards_(history_df, spread_coef, lot_coef, lot_size, deal_length, history_step_size)

        buy_ema = np.mean(buy_abs_rewards)
        ema_buy_rewards = []
        for i in range( len(buy_abs_rewards) ):
            buy_reward = buy_abs_rewards[i] / (buy_ema + 1.0)
            buy_ema = alpha * buy_reward + (1.0 - alpha) * (buy_ema)
            ema_buy_rewards.append( buy_reward )
        ema_buy_rewards = np.array( ema_buy_rewards )

        sell_ema = np.mean(sell_abs_rewards)
        ema_sell_rewards = []
        for i in range( len(sell_abs_rewards) ):
            sell_reward = sell_abs_rewards[i] / (sell_ema + 1.0)
            sell_ema = alpha * sell_reward + (1.0 - alpha) * (sell_ema)
            ema_sell_rewards.append( sell_reward )
        ema_sell_rewards = np.array(ema_sell_rewards)

        ema_buy_rewards = self.clean_outliers( ema_buy_rewards )
        ema_sell_rewards = self.clean_outliers( ema_sell_rewards )

        all_rewards = np.hstack( [ema_buy_rewards, ema_sell_rewards] )

        self.scaler.fit( all_rewards.reshape(-1, 1) )

        return self

    def transform(self, reward):

        reward_sign = np.sign( reward )
        abs_reward = np.abs( reward )

        transformed_reward = reward_sign * self.scaler.transform( np.array(abs_reward).reshape((-1, 1)) )[0][0]

        return transformed_reward

    def collect_rewards_(self, history_df, spread_coef, lot_coef, lot_size,
                         deal_length, history_step_size):

        history_price = history_df.values
        features_dict = {}
        for i, feat in zip( range(len(history_df.columns)), history_df.columns ):
            features_dict[feat] = i

        abs_buy_rewards = []
        abs_sell_rewards = []
        for i in tqdm(range(0, len(history_df) - deal_length - 1, history_step_size), desc="Collecting rewards"):
            open_point_open_price = history_price[i][features_dict["open"]]
            open_point_spread = history_price[i][features_dict["spread"]]

            close_point_open_price = history_price[i + deal_length][features_dict["open"]]
            close_point_spread = history_price[i + deal_length][features_dict["spread"]]

            buy_reward = (close_point_open_price - (open_point_open_price + spread_coef * open_point_spread)) * lot_coef * lot_size
            sell_reward = (open_point_open_price - (close_point_open_price + spread_coef * close_point_spread)) * lot_coef * lot_size

            abs_buy_reward = np.abs( buy_reward )
            abs_sell_reward = np.abs( sell_reward )

            abs_buy_rewards.append( abs_buy_reward )
            abs_sell_rewards.append( abs_sell_reward )

        abs_buy_rewards = np.array( abs_buy_rewards )
        abs_sell_rewards = np.array( abs_sell_rewards )

        return abs_buy_rewards, abs_sell_rewards

    def clean_outliers(self, rewards, quantile = 0.99):

        q_val = np.quantile(np.abs(rewards), q=quantile)
        rewards = rewards[ np.abs(rewards) <= q_val ]

        return rewards