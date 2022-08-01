
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


class OnFlySupervisedSampleGenerator():
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
        #del self.common_targets
        pass

    def get_sample(self, history_step_id):

        x = self.base_generator.get_features(self.history_price_array, self.price_feature_names_dict, history_step_id)
        y = self.step_id_target_mapping[history_step_id]

        """start_point = self.common_step_ids[0]
        y_id = history_step_id - start_point

        if y_id < 0:
            raise Exception("Broken history ids mapping. {}_{} sampler".format(self.symbol, self.timeframe))
        check_x = self.common_step_ids[y_id]
        if check_x != history_step_id:
            raise Exception("Broken history ids mapping. {}_{} sampler. {}: {}".format(self.symbol, self.timeframe, y_id, check_x))

        y = self.common_targets[ y_id ]"""

        return x, y

    def get_train_val_split(self, test_size=0.2, shuffle=True):

        x_train, x_val, y_train, y_val = train_test_split( self.common_step_ids, self.common_targets,
                                                           random_state=45, shuffle=shuffle, test_size=test_size)
        return x_train, x_val, y_train, y_val


    def fit(self, mod_df, composite_env, norm_ema=0, smooth_ema=10, start_hold_proba=0.9, end_hold_proba=0.95, n_estimates=100, show_plots=False):

        self.history_price_array = composite_env.history_price
        self.price_feature_names_dict = composite_env.price_feature_names_dict

        buyer_step_ids, buyer_targets, \
        seller_step_ids, seller_targets = \
            self.generate_stochastic_estimates_(mod_df, composite_env,
                                                start_hold_proba, end_hold_proba,
                                                n_estimates)

        common_step_ids, common_targets = self.make_common_targets_(buyer_step_ids, buyer_targets,
                                                                    seller_step_ids, seller_targets)


        common_step_ids, common_targets = self.preprocess_targets_( common_step_ids, common_targets, norm_ema, smooth_ema, show_plots=show_plots )

        self.common_step_ids = common_step_ids
        self.common_targets = common_targets

        return self

    def generate_stochastic_estimates_(self, mod_df, composite_env, start_hold_proba, end_hold_proba, n_estimates):

        dfs = [deepcopy(mod_df), deepcopy(mod_df)]
        symbols = [deepcopy(self.symbol), deepcopy(self.symbol)]
        timeframes = [deepcopy(self.timeframe), deepcopy(self.timeframe)]
        composite_envs = [deepcopy(composite_env), deepcopy(composite_env)]
        closer_types = ["buyer", "seller"]
        feature_list = [deepcopy(self.feature_list), deepcopy(self.feature_list)]
        n_points = [deepcopy(self.n_points), deepcopy(self.n_points)]
        flat_stack = [deepcopy(self.flat_stack), deepcopy(self.flat_stack)]
        start_hold_probas = [deepcopy(start_hold_proba), deepcopy(start_hold_proba)]
        end_hold_probas = [deepcopy(end_hold_proba), deepcopy(end_hold_proba)]
        estimate_counts = [deepcopy(n_estimates), deepcopy(n_estimates)]

        stochastic_estimates = Parallel(n_jobs=2)(delayed(generate_stochastic_estimates_)
                                                   (df, ss, tf, ce, ct, fl, nps, fs, shp, ehp, ec)
                                                   for df, ss, tf, ce, ct, fl, nps, fs, shp, ehp, ec in
                                                   zip(dfs, symbols, timeframes,
                                                       composite_envs, closer_types,
                                                       feature_list, n_points, flat_stack,
                                                       start_hold_probas, end_hold_probas, estimate_counts))
        buyer_step_ids, buyer_targets = stochastic_estimates[0][0], stochastic_estimates[0][1]
        seller_step_ids, seller_targets = stochastic_estimates[1][0], stochastic_estimates[1][1]

        return buyer_step_ids, buyer_targets, seller_step_ids, seller_targets

    def show_plots(self, reward_line_plot=True, reward_dist_plot=True):

        y = self.common_targets

        if reward_line_plot:
            axis_x = [i for i in range(len(y))]
            plt.plot( axis_x, y[:, 0], c="g" )
            plt.plot( axis_x, y[:, 1], c="r" )
            plt.title(self.symbol + "_" + self.timeframe)
            plt.show()

        if reward_dist_plot:
            sns.displot(y.reshape((y.shape[0] * y.shape[1], 1)))
            plt.title(self.symbol + "_" + self.timeframe)
            plt.show()

        pass

    def preprocess_targets_(self, common_step_ids, common_targets, norm_ema=0, smooth_ema=10, show_plots=False):

        x = common_step_ids
        y = common_targets

        x, y = self.ema_normalize_targets_(x, y, n=norm_ema)
        x, y = self.smooth_targets_( x, y, n=smooth_ema )

        # targets for scaler without outliers
        buyer_quantile = np.quantile(np.abs(y[:, 0]), q=0.99)
        seller_quantile = np.quantile(np.abs(y[:, 1]), q=0.99)
        scale_y_buyer = y[np.abs(y[:, 0]) < buyer_quantile, 0]
        scale_y_seller = y[np.abs(y[:, 1]) < seller_quantile, 1]
        scale_y = np.hstack( [scale_y_buyer, scale_y_seller] )
        scale_y = scale_y.reshape((scale_y.shape[-1], 1))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(scale_y)
        y[:, 0] = scaler.transform(y[:, 0].reshape((-1, 1))).reshape((-1,))
        y[:, 1] = scaler.transform(y[:, 1].reshape((-1, 1))).reshape((-1,))

        # remove close to zero samples
        #x = x[(y[:, 0] < -0.25) | (y[:, 0] > 0.05) ]
        #y = y[(y[:, 0] < -0.25) | (y[:, 0] > 0.05)]
        #x = x[(y[:, 1] < -0.25) | (y[:, 1] > 0.05)]
        #y = y[(y[:, 1] < -0.25) | (y[:, 1] > 0.05)]

        return x, y

    def ema_normalize_targets_(self, x, y, n=20):

        if n <= 0:
            return x, y

        alpha = 1.0 / float(n)

        buyer_ema = sum(y[:n, 0]) / float(n)
        seller_ema = sum(y[:n, 1]) / float(n)

        x = x[n:]
        y = y[n:]
        for i in range(len(y)):
            y[i][0] = y[i][0] / (np.abs(buyer_ema) + 1.0)
            y[i][1] = y[i][1] / (np.abs(seller_ema) + 1.0)
            buyer_ema = alpha * y[i][0] + (1.0 - alpha) * buyer_ema
            seller_ema = alpha * y[i][1] + (1.0 - alpha) * seller_ema

        return x, y

    def smooth_targets_(self, x, y, n=20):

        if n <= 0:
            return x, y

        alpha = 1.0 / float(n)

        buyer_ema = sum( y[:n, 0] ) / float( n )
        seller_ema = sum( y[:n, 1] ) / float( n )

        x = x[n:]
        y = y[n:]
        for i in range( len(y) ):
            buyer_ema = alpha * y[i][0] + (1.0 - alpha) * buyer_ema
            seller_ema = alpha * y[i][1] + (1.0 - alpha) * seller_ema
            y[i][0] = buyer_ema
            y[i][1] = seller_ema

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

def generate_stochastic_estimates_(mod_df, symbol, timeframe, composite_env, closer_type, feature_list, n_points, flat_stack,
                                   start_hold_proba=0.2, end_hold_proba=0.95, n_estimates=100):

        history_df = mod_df.copy()
        agent = StochasticCloser()
        agent_env = composite_env.state_dict["{}".format(closer_type)]
        agent_env.feature_generator = NoneGenerator(feature_list, n_points=n_points, flat_stack=flat_stack)

        history_price_array = composite_env.history_price
        price_feature_names_dict = composite_env.price_feature_names_dict

        steps_i = []
        close_rewards = []
        start_point = composite_env.get_start_point()

        start_hold_proba = start_hold_proba
        start_close_proba = 1.0 - start_hold_proba
        end_hold_proba = end_hold_proba
        n_estimates = n_estimates
        proba_increment = (end_hold_proba - start_hold_proba) / n_estimates

        for i in tqdm(range(start_point, len(history_df) - 2), desc="Collecting {} targets for {}_{}".format( closer_type, symbol, timeframe )):

            steps_i.append(i)
            agent_env.set_open_point(i)
            state_descriptor = agent_env.step(history_price_array, price_feature_names_dict, i, 0)

            state_value_scores = []
            current_hold_proba = start_hold_proba
            current_close_proba = start_close_proba
            for estimate_i in range(n_estimates):
                j = i
                while state_descriptor.action != 1:
                    j += 1
                    if j == len(history_df) - 2:
                        break

                    action = agent.get_action_head_only(current_hold_proba, current_close_proba)
                    state_descriptor = agent_env.step(history_price_array, price_feature_names_dict, j, action)

                if j == len(history_df) - 2:
                    break

                reward = state_descriptor.reward_dict[1]
                state_descriptor = agent_env.step(history_price_array, price_feature_names_dict, i, 0)
                state_value_scores.append(reward)
                current_hold_proba += proba_increment
                current_close_proba -= proba_increment

            #state_value_score = np.mean(state_value_scores)
            state_value_score = np.median(state_value_scores)
            #state_value_score = gmean(state_value_scores)
            #state_value_score = hmean(state_value_scores)
            close_rewards.append(state_value_score)

        steps_i = np.array( steps_i )
        close_rewards = np.array( close_rewards )

        return steps_i, close_rewards
