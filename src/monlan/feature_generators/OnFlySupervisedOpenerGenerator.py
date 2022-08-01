
import os
import numpy as np
from scipy.stats import gmean, hmean
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from joblib import Parallel, delayed

from src.monlan.modular_agents.StochasticCloser import StochasticCloser
from src.monlan.feature_generators.NoneGenerator import NoneGenerator
from src.monlan.feature_generators.FeatGen_HistoryEmbedder import FeatGen_HistoryEmbedder


class OnFlySupervisedOpenerGenerator():
    def __init__(self, symbol, timeframe, base_generator):

        self.base_generator = base_generator
        self.symbol = symbol
        self.timeframe = timeframe


        self.feature_list = base_generator.feature_list
        self.feature_list_size = len( base_generator.feature_list )
        self.n_points = base_generator.n_points
        self.flat_stack = base_generator.flat_stack
        self.feature_shape = base_generator.feature_shape

        self.history_price_array = None
        self.price_feature_names_dict = None
        self.common_step_ids = None
        self.common_targets = None

        pass

    def remake_history_step_target_mapping(self):
        self.step_id_target_mapping = {}
        for i in range(len(self.common_step_ids)):
            self.step_id_target_mapping[self.common_step_ids[i]] = self.common_targets[i]
        pass

    def get_sample(self, history_step_id):

        x = self.base_generator.get_features(self.history_price_array, self.price_feature_names_dict, history_step_id)
        y = self.step_id_target_mapping[history_step_id]

        return x, y

    def get_train_val_split(self, test_size=0.2, shuffle=True):

        x_train, x_val, y_train, y_val = train_test_split( self.common_step_ids, self.common_targets,
                                                           random_state=45, shuffle=shuffle, test_size=test_size)
        return x_train, x_val, y_train, y_val


    def fit(self, mod_df, composite_env, trained_closer, norm_ema=0, smooth_ema=10, show_plots=False):

        self.history_price_array = composite_env.history_price
        self.price_feature_names_dict = composite_env.price_feature_names_dict

        buyer_step_ids, buyer_targets, \
        seller_step_ids, seller_targets = \
            self.collect_trained_closer_targets_(mod_df, composite_env, trained_closer)

        common_step_ids, common_targets = self.make_common_targets_(buyer_step_ids, buyer_targets,
                                                                    seller_step_ids, seller_targets)

        common_step_ids, common_targets = self.add_hold_targets_(common_step_ids, common_targets, composite_env)

        common_step_ids, common_targets = self.preprocess_targets_( common_step_ids, common_targets, norm_ema, smooth_ema, show_plots=show_plots )

        self.common_step_ids = common_step_ids
        self.common_targets = common_targets

        return self

    def add_hold_targets_(self, common_step_ids, common_targets, agent_env):

        extended_targets = []
        lot_size = agent_env.lot_size
        lot_coef = agent_env.lot_coef
        open_prices = self.history_price_array[:, self.price_feature_names_dict["open"]]
        close_prices = self.history_price_array[:, self.price_feature_names_dict["close"]]
        for i in range(len(common_step_ids)):
            step_id = common_step_ids[i]
            hold_target = abs(open_prices[step_id] - close_prices[step_id]) * lot_coef * lot_size
            extended_target = [common_targets[i][0], hold_target, common_targets[i][1]]
            extended_targets.append( extended_target )
        extended_targets = np.array( extended_targets )

        return common_step_ids, extended_targets

    def collect_trained_closer_targets_(self, mod_df, composite_env, trained_closer):

        dfs = [deepcopy(mod_df), deepcopy(mod_df)]
        symbols = [deepcopy(self.symbol), deepcopy(self.symbol)]
        timeframes = [deepcopy(self.timeframe), deepcopy(self.timeframe)]
        composite_envs = [deepcopy(composite_env), deepcopy(composite_env)]
        closer_types = ["buyer", "seller"]
        trained_closers = [deepcopy(trained_closer), deepcopy(trained_closer)]

        closer_targets = Parallel(n_jobs=2)(delayed(generate_closer_targets_)
                                                   (df, ss, tf, ce, ct, tc)
                                                   for df, ss, tf, ce, ct, tc in
                                                   zip(dfs, symbols, timeframes,
                                                       composite_envs, closer_types,
                                                       trained_closers))
        buyer_step_ids, buyer_targets = closer_targets[0][0], closer_targets[0][1]
        seller_step_ids, seller_targets = closer_targets[1][0], closer_targets[1][1]

        return buyer_step_ids, buyer_targets, seller_step_ids, seller_targets

    def show_plots(self, reward_line_plot=True, reward_dist_plot=True):

        y = self.common_targets

        if reward_line_plot:
            axis_x = [i for i in range(len(y))]
            plt.plot(axis_x, y[:, 0], c="g")
            plt.plot(axis_x, y[:, 2], c="r")
            plt.plot(axis_x, y[:, 1], c="b")
            plt.show()

        if reward_dist_plot:
            sns.displot(y.reshape((y.shape[0] * y.shape[2], 1)))
            plt.title(self.symbol + "_" + self.timeframe)
            plt.show()

        pass

    def preprocess_targets_(self, common_step_ids, common_targets, norm_ema=0, smooth_ema=10, show_plots=False):

        x = common_step_ids
        y = common_targets

        x, y = self.ema_normalize_targets_(x, y, n=norm_ema)
        x, y = self.smooth_targets_(x, y, n=smooth_ema)

        # modify hold reward
        y[:, 1] = 1.16 * y[:, 1]

        # targets for scaler without outliers
        buyer_quantile = np.quantile(np.abs(y[:, 0]), q=0.99)
        seller_quantile = np.quantile(np.abs(y[:, 2]), q=0.99)
        scale_y_buyer = y[np.abs(y[:, 0]) < buyer_quantile, 0]
        scale_y_seller = y[np.abs(y[:, 2]) < seller_quantile, 2]
        scale_y = np.hstack([scale_y_buyer, scale_y_seller])
        scale_y = scale_y.reshape((scale_y.shape[-1], 1))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(scale_y)
        for i in range(y.shape[1]):
            y[:, i] = scaler.transform(y[:, i].reshape((-1, 1))).reshape((-1,))

        return x, y

    def ema_normalize_targets_(self, x, y, n=0):

        if n <= 0:
            return x, y

        alpha = 1.0 / float(n)

        emas = []
        target_width = y.shape[1]
        for i in range(target_width):
            emas.append(sum(y[:n, i]) / float(n))

        x = x[n:]
        y = y[n:]
        for i in range(len(y)):
            for j in range(target_width):
                y[i][j] = y[i][j] / (np.abs(emas[j]) + 1.0)
                emas[j] = alpha * y[i][j] + (1.0 - alpha) * emas[j]

        return x, y

    def smooth_targets_(self, x, y, n=10):

        if n <= 0:
            return x, y

        alpha = 1.0 / float(n)

        emas = []
        target_width = y.shape[1]
        for i in range(target_width):
            emas.append(sum(y[:n, i]) / float(n))

        x = x[n:]
        y = y[n:]
        for i in range(len(y)):
            for j in range(target_width):
                emas[j] = alpha * y[i][j] + (1.0 - alpha) * emas[j]
                y[i][j] = emas[j]

        return x, y

    def make_common_targets_(self, buyer_step_ids, buyer_targets, seller_step_ids, seller_targets):

        min_targets_length = min( len(buyer_step_ids), len(seller_step_ids) )

        common_step_ids = []
        common_targets = []
        for i in range( min_targets_length ):
            if buyer_step_ids[i] != seller_step_ids[i]:
                raise Exception("History step ids are different!")

            if np.isnan( np.sum([buyer_targets[i], seller_targets[i]]) ):
                continue

            common_step_ids.append( buyer_step_ids[i] )
            common_targets.append( [buyer_targets[i], seller_targets[i]] )

        common_step_ids = np.array( common_step_ids )
        common_targets = np.array( common_targets )

        return common_step_ids, common_targets

    def plot_ohlc_targets(self, candle_count):

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

        start_point = self.common_step_ids[0]
        hk_open = self.history_price_array[start_point: start_point + candle_count, self.price_feature_names_dict["open"]]
        hk_close = self.history_price_array[start_point: start_point + candle_count, self.price_feature_names_dict["close"]]
        hk_high = self.history_price_array[start_point: start_point + candle_count, self.price_feature_names_dict["high"]]
        hk_low = self.history_price_array[start_point: start_point + candle_count, self.price_feature_names_dict["low"]]

        for i in tqdm(range(hk_open.shape[0])):
            open = hk_open[i]
            close = hk_close[i]
            if close > open:
                color = "green"
                ax[0].plot([i, i], [open, close], c=color, linewidth=2.0)
                up_line = [hk_close[i], hk_high[i]]
                down_line = [hk_open[i], hk_low[i]]
                ax[0].plot([i, i], up_line, c="black", linewidth=0.5)
                ax[0].plot([i, i], down_line, c="black", linewidth=0.5)
            else:
                color = "red"
                ax[0].plot([i, i], [close, open], c=color, linewidth=2.0)
                up_line = [hk_open[i], hk_high[i]]
                down_line = [hk_close[i], hk_low[i]]
                ax[0].plot([i, i], up_line, c="black", linewidth=0.5)
                ax[0].plot([i, i], down_line, c="black", linewidth=0.5)

        selected_targets = self.common_targets[:candle_count]
        x_range = [i for i in range(candle_count)]
        ax[1].plot( x_range, selected_targets[:, 0], color="g" )
        ax[1].plot( x_range, selected_targets[:, 1], color="r" )

        plt.show()

        pass

def generate_closer_targets_(mod_df, symbol, timeframe, composite_env, closer_type, trained_closer):

        history_df = mod_df.copy()
        agent = trained_closer
        agent.name = closer_type
        agent_env = composite_env.state_dict["{}".format(closer_type)]

        agent_feat_generator = FeatGen_HistoryEmbedder(agent_env.feature_generator, agent.model)
        history_price_array = composite_env.history_price
        price_feature_names_dict = composite_env.price_feature_names_dict
        start_point = composite_env.get_start_point()
        agent_feat_generator.fit(history_price_array, price_feature_names_dict, start_point, batch_size=64)
        agent_env.feature_generator = agent_feat_generator

        steps_i = []
        close_rewards = []
        start_point = composite_env.get_start_point()


        for i in tqdm(range(start_point, len(history_df) - 2), desc="Collecting {} targets for {}_{} opener".format( closer_type, symbol, timeframe )):

            open_embedding = agent_feat_generator.get_features(history_price_array, price_feature_names_dict, i)
            action = agent.get_action_head_only(open_embedding)

            agent_env.set_open_point(i)
            state_descriptor = agent_env.step(history_price_array, price_feature_names_dict, i, action)

            j = i
            while state_descriptor.action != 1:
                j += 1
                if j == len(history_df) - 2:
                    break

                next_obs = agent_feat_generator.get_features(history_price_array, price_feature_names_dict, j)
                action = agent.get_action_head_only(next_obs)
                state_descriptor = agent_env.step(history_price_array, price_feature_names_dict, j, action)

            if j == len(history_df) - 2:
                break

            reward = state_descriptor.reward_dict[1]
            steps_i.append(i)
            close_rewards.append(reward)

        steps_i = np.array( steps_i )
        close_rewards = np.array( close_rewards )

        return steps_i, close_rewards
