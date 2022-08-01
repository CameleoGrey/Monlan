
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

class RewardTransformer():
    def __init__(self):

        self.power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        pass

    def fit(self, history_df, spread_coef, lot_coef, lot_size, window_width=20):

        collected_rewards = self.collect_rewards_(history_df, spread_coef, lot_coef, lot_size, window_width)
        #collected_rewards = self.clean_outliers(collected_rewards, quantile=0.99)
        self.scaler.fit( collected_rewards.reshape(-1, 1) )
        #collected_rewards = self.scaler.transform( collected_rewards.reshape(-1, 1) )
        #sns.displot(collected_rewards)
        #plt.show()

        return self

    def transform(self, reward):

        transformed_reward = self.scaler.transform( np.array(reward).reshape((-1, 1)) )[0][0]

        return transformed_reward

    def collect_rewards_(self, history_df, spread_coef, lot_coef, lot_size, window_width):

        history_price = history_df.values
        features_dict = {}
        for i, feat in zip( range(len(history_df.columns)), history_df.columns ):
            features_dict[feat] = i

        collected_rewards = []
        for i in tqdm(range(len(history_df) - window_width - 1), desc="Collecting rewards"):
            open_point_open_price = history_price[i][features_dict["open"]]
            open_point_spread = history_price[i][features_dict["spread"]]

            for j in range(i+1, i+1+window_width):
                close_point_open_price = history_price[j][features_dict["open"]]
                close_point_spread = history_price[j][features_dict["spread"]]

                buy_reward = (close_point_open_price - (open_point_open_price + spread_coef * open_point_spread)) * lot_coef * lot_size
                sell_reward = (open_point_open_price - (close_point_open_price + spread_coef * close_point_spread)) * lot_coef * lot_size

                collected_rewards.append( buy_reward )
                collected_rewards.append( sell_reward )
        collected_rewards = np.array( collected_rewards )

        return collected_rewards

    def clean_outliers(self, rewards, quantile = 0.99):

        q_val = np.quantile(np.abs(rewards), q=quantile)
        rewards = rewards[ np.abs(rewards) <= q_val ]

        return rewards