import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset

from src.monlan.modular_agents.tcn.tcn import *
from src.monlan.utils.save_load import *

import os
import random
import numpy as np
from collections import deque
from scipy.special import softmax
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import smogn

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.float32(x)
        self.y = np.float32(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_i = self.x[idx]
        x_i = x_i.reshape( (x_i.shape[1], x_i.shape[2]) )
        y_i = self.y[idx]
        return x_i, y_i

class DistilledHybridMirrorOpener:
    def __init__(self, mirror_closer):

        self.name = "opener"

        self.state_size = None
        self.action_size = 3
        self.epsilon = 0.0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = mirror_closer.model
        self.regression_model = None

    def set_name(self, name):
        self.name = name
        pass

    def get_name(self):
        return self.name

    def fit(self, buyer_samples, seller_samples, ema_norm_n = 20, smooth_target_n = 20, test_size=0.33, show_plots=False):

        train_x, val_x, train_y, val_y = self.build_train_val_dataset_( buyer_samples, seller_samples,
                                                                        ema_norm_n, smooth_target_n, test_size=test_size,
                                                                        show_plots=show_plots )

        """regression_model = CatBoostRegressor( n_estimators=20000,
                                              max_depth=4,
                                              #loss_function="Quantile",
                                              loss_function="MultiRMSE",
                                              thread_count=8,
                                              #task_type="GPU"
                                              )
        regression_model.fit(train_x, train_y, eval_set=(val_x, val_y),
                             use_best_model=True, verbose=True, early_stopping_rounds=200)"""

        """regression_model = MultiOutputRegressor(CatBoostRegressor( n_estimators=10000,
                                              max_depth=4,
                                              #loss_function="Quantile",
                                              loss_function="RMSE",
                                              #thread_count=8,
                                              task_type="GPU"
                                              ),
                                              n_jobs=1)
        regression_model.fit( train_x, train_y  )"""

        regression_models = []
        for i in range( train_y.shape[1] ):
            partial_regression_model = CatBoostRegressor( n_estimators=10000,
                                                          max_depth=4,
                                                          #loss_function="Quantile:alpha=0.5",
                                                          loss_function="RMSE",
                                                          thread_count=8,
                                                          #task_type="GPU",
                                                          early_stopping_rounds=100,
                                                          use_best_model=True,
                                                          #learning_rate=0.001
                                                        )

            partial_regression_model.fit( train_x, train_y[:, i], eval_set=(val_x, val_y[:, i]) )

            pred_y = partial_regression_model.predict(val_x)
            score = mean_squared_error(val_y[:, i], pred_y)
            print("Partial validation error for {} output: {}".format(i, score))

            regression_models.append( partial_regression_model )
        regression_model = MultiOutputRegressor(None)
        regression_model.estimators_ = regression_models
        regression_model.feature_names_in_ = ["{}".format(i) for i in range(train_y.shape[1])]
        regression_model.n_features_in_ = train_x.shape[1]


        pred_y = regression_model.predict( val_x )
        score = mean_squared_error( val_y, pred_y )
        print("Validation error: {}".format(score))

        self.regression_model = regression_model

        return self

    def build_train_val_dataset_(self, buyer_samples, seller_samples, ema_norm_n=20, smooth_target_n=20, test_size=0.33, show_plots=False):

        buyer_steps = buyer_samples["id"]
        buyer_raw_x = buyer_samples["x_raw"]
        buyer_embedded_x = buyer_samples["x_embedded"]
        buyer_y = buyer_samples["y"]

        seller_steps = seller_samples["id"]
        seller_raw_x = seller_samples["x_raw"]
        seller_embedded_x = seller_samples["x_embedded"]
        seller_y = seller_samples["y"]

        hold_targets = self.build_hold_targets_(buyer_samples, lot_coef=100000, lot_size=0.1)

        common_x = []
        common_y = []
        min_dataset_len = min(len(buyer_steps), len(seller_steps))
        for i in tqdm(range(min_dataset_len), desc="Building common dataset"):
            if buyer_steps[i] != seller_steps[i]:
                raise ValueError("Different step_i")
            if buyer_raw_x[i][0][0][0][0] != seller_raw_x[i][0][0][0][0]:
                raise ValueError("Different obs")

            target = [buyer_y[i], hold_targets[i], seller_y[i]]
            target_sum = np.sum(target)
            if np.isnan(target_sum):
                continue
            common_y.append(target)
            common_x.append(buyer_embedded_x[i])

        common_x = np.array(common_x)
        common_y = np.array(common_y)

        x = common_x
        y = common_y

        x, y = self.ema_normalize_targets_(x, y, n=ema_norm_n)
        x, y = self.smooth_targets_(x, y, n=smooth_target_n)

        # modify hold reward
        y[:, 1] = 1.0 * y[:, 1]

        if show_plots:
            axis_x = [i for i in range(len(y))]
            plt.plot(axis_x, y[:, 0], c="g")
            plt.plot(axis_x, y[:, 2], c="r")
            plt.plot(axis_x, y[:, 1], c="b")
            plt.show()

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

        # remove close to zero samples
        # x = x[(y[:, 0] < -0.25) | (y[:, 0] > 0.05) ]
        # y = y[(y[:, 0] < -0.25) | (y[:, 0] > 0.05)]
        # x = x[(y[:, 1] < -0.25) | (y[:, 1] > 0.05)]
        # y = y[(y[:, 1] < -0.25) | (y[:, 1] > 0.05)]

        # hold distribution plot
        if show_plots:
            sns.displot(y[:, 1].reshape((y.shape[0], 1)))
            plt.show()
            # buyer/seller distribution plot
            bs_vals = np.vstack([y[:, 0], y[:, 2]])
            sns.displot(bs_vals.reshape((bs_vals.shape[0] * bs_vals.shape[1], 1)))
            plt.show()

        train_x, val_x, train_y, val_y = train_test_split(x, y, shuffle=True, test_size=test_size, random_state=45)

        print("Train dataset size: {}".format(len(train_y)))
        print("Val dataset size: {}".format(len(val_y)))

        return train_x, val_x, train_y, val_y

    def build_hold_targets_(self, samples, lot_coef=100000, lot_size=0.1):
        price_vals = [samples["open"], samples["high"], samples["low"], samples["close"]]
        hold_targets = []
        for i in range(len(price_vals[0])):
            hold_target = abs(price_vals[3][i] - price_vals[0][i]) * lot_coef * lot_size
            #hold_target = abs(price_vals[1][i] - price_vals[2][i]) * lot_coef * lot_size
            hold_targets.append(hold_target)
        hold_targets = np.array(hold_targets)
        return hold_targets

    def ema_normalize_targets_(self, x, y, n=40):
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

    def smooth_targets_(self, x, y, n=40):

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

    def get_action_head_only(self, state):

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
            state = state.to(self.device)

        with torch.no_grad():
            state_embedding = self.encoder.linear(state).cpu().detach().numpy()[0]
            q_value = self.regression_model.predict(state_embedding)
        q_value = q_value[0]

        # modify hold
        q_value[1] = 1.0 * q_value[1]

        greed_action_id = np.argmax(q_value)

        return greed_action_id

    def get_action(self, state):

        # 1d conv
        state = state.reshape((1, state.shape[1], state.shape[2]))

        # 2d conv
        # state = state.reshape((1, 1, state.shape[1], state.shape[2]))

        state = torch.Tensor(state).to(self.device)

        with torch.no_grad():
            state_embedding = self.encoder.get_embeddings(state).cpu().detach().numpy()[0]

            # fix for sklearn regressors
            state_embedding = state_embedding.reshape((1, -1))
            q_value = self.regression_model.predict(state_embedding)[0]

        # modify hold
        q_value[1] = 1.0 * q_value[1]

        greed_action_id = np.argmax(q_value)

        return greed_action_id