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

class DistilledMultiHybridMirrorCloser:
    def __init__(self, buyer_rl_agent, seller_rl_agent):

        self.name = None

        self.state_size = None
        self.action_size = 2
        self.epsilon = 0.0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoders = [buyer_rl_agent.model, seller_rl_agent.model]
        self.regression_model = None

    def set_name(self, name):
        self.name = name
        pass

    def get_name(self):
        return self.name

    def fit(self, buyer_samples, seller_samples, ema_norm_n = 20, smooth_target_n = 20, test_size=0.33):

        train_x, val_x, train_y, val_y = self.build_train_val_dataset_( buyer_samples, seller_samples, ema_norm_n, smooth_target_n, test_size=test_size )

        """regression_model = CatBoostRegressor( n_estimators=20000,
                                              max_depth=4,
                                              #loss_function="Quantile",
                                              loss_function="MultiRMSE",
                                              thread_count=8,
                                              #task_type="GPU"
                                              )
        regression_model.fit(train_x, train_y, eval_set=(val_x, val_y),
                             use_best_model=True, verbose=True, early_stopping_rounds=200)"""

        regression_model = MultiOutputRegressor(CatBoostRegressor( n_estimators=4000,
                                              max_depth=4,
                                              #loss_function="Quantile",
                                              loss_function="RMSE",
                                              #thread_count=8,
                                              task_type="GPU"
                                              ),
                                              n_jobs=1)
        regression_model.fit( train_x, train_y )
        pred_y = regression_model.predict( val_x )
        score = mean_squared_error( val_y, pred_y )
        print("Validation error: {}".format(score))

        self.regression_model = regression_model

        return self

    def build_train_val_dataset_(self, buyer_samples, seller_samples, ema_norm_n = 20, smooth_target_n = 20, test_size=0.33):

        buyer_steps = buyer_samples["id"]
        buyer_raw_x = buyer_samples["x_raw"]
        buyer_embedded_x = buyer_samples["x_embedded"]
        buyer_y = buyer_samples["y"]

        seller_steps = seller_samples["id"]
        seller_raw_x = seller_samples["x_raw"]
        seller_embedded_x = seller_samples["x_embedded"]
        seller_y = seller_samples["y"]

        common_x = []
        common_y = []
        min_dataset_len = min(len(buyer_steps), len(seller_steps))
        for i in tqdm(range(min_dataset_len), desc="Building common dataset"):
            if buyer_steps[i] != seller_steps[i]:
                raise ValueError("Different step_i")
            if buyer_raw_x[i][0][0][0][0] != seller_raw_x[i][0][0][0][0]:
                raise ValueError("Different obs")

            united_x = np.hstack([buyer_embedded_x[i], seller_embedded_x[i]])
            common_x.append(united_x)
            target = [buyer_y[i], seller_y[i]]
            common_y.append(target)

        common_x = np.array(common_x)
        common_y = np.array(common_y)

        x = common_x
        y = common_y

        x, y = self.ema_normalize_targets_(x, y, n=ema_norm_n)
        x, y = self.smooth_targets_(x, y, n=smooth_target_n)

        # axis_x = [i for i in range(len(y))]
        # plt.plot( axis_x, y[:, 0], c="g" )
        # plt.plot( axis_x, y[:, 1], c="r" )
        # plt.show()

        # targets for scaler without outliers
        buyer_quantile = np.quantile(np.abs(y[:, 0]), q=0.99)
        seller_quantile = np.quantile(np.abs(y[:, 1]), q=0.99)
        scale_y_buyer = y[np.abs(y[:, 0]) < buyer_quantile, 0]
        scale_y_seller = y[np.abs(y[:, 1]) < seller_quantile, 1]
        scale_y = np.hstack([scale_y_buyer, scale_y_seller])
        scale_y = scale_y.reshape((scale_y.shape[-1], 1))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(scale_y)
        y[:, 0] = scaler.transform(y[:, 0].reshape((-1, 1))).reshape((-1,))
        y[:, 1] = scaler.transform(y[:, 1].reshape((-1, 1))).reshape((-1,))

        # remove close to zero samples
        x = x[(y[:, 0] < -0.25) | (y[:, 0] > 0.05) ]
        y = y[(y[:, 0] < -0.25) | (y[:, 0] > 0.05)]
        x = x[(y[:, 1] < -0.25) | (y[:, 1] > 0.05)]
        y = y[(y[:, 1] < -0.25) | (y[:, 1] > 0.05)]

        sns.displot( y.reshape((y.shape[0] * y.shape[1], 1)) )
        plt.show()

        train_x, val_x, train_y, val_y = train_test_split(x, y, shuffle=True, test_size=test_size, random_state=45)

        print("Train dataset size: {}".format(len(train_y)))
        print("Val dataset size: {}".format(len(val_y)))

        return train_x, val_x, train_y, val_y

    def ema_normalize_targets_(self, x, y, n=40):
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

    def smooth_targets_(self, x, y, n=40):

        alpha = 1.0 / float(n)

        buyer_ema = sum(y[:n, 0]) / float(n)
        seller_ema = sum(y[:n, 1]) / float(n)

        x = x[n:]
        y = y[n:]
        for i in range(len(y)):
            buyer_ema = alpha * y[i][0] + (1.0 - alpha) * buyer_ema
            seller_ema = alpha * y[i][1] + (1.0 - alpha) * seller_ema
            y[i][0] = buyer_ema
            y[i][1] = seller_ema

        return x, y

    def get_action_head_only(self, state):

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
            state = state.to(self.device)

        with torch.no_grad():
            buyer_state_embedding = self.encoders[0].linear(state).cpu().detach().numpy()[0]
            seller_state_embedding = self.encoders[1].linear(state).cpu().detach().numpy()[0]
            buyer_state_value = self.regression_model[0].predict(buyer_state_embedding)
            seller_state_value = self.regression_model[1].predict(seller_state_embedding)
            q_value = [buyer_state_value, seller_state_value]
        q_value = q_value[0]
        if self.name == "buyer":
            q_value[0], q_value[1] = q_value[0], q_value[1]
        elif self.name == "seller":
            q_value[1], q_value[0] = q_value[0], q_value[1]
        else:
            raise ValueError("Name of mirror agent must be specified (buyer, seller) before using")

        greed_action_id = np.argmax(q_value)

        return greed_action_id

    def get_action(self, state):

        # 1d conv
        state = state.reshape((1, state.shape[1], state.shape[2]))

        # 2d conv
        # state = state.reshape((1, 1, state.shape[1], state.shape[2]))

        state = torch.Tensor(state).to(self.device)

        with torch.no_grad():
            buyer_state_embedding = self.encoders[0].get_embeddings(state).cpu().detach().numpy()[0]
            seller_state_embedding = self.encoders[1].get_embeddings(state).cpu().detach().numpy()[0]
            state_embedding = np.hstack( [buyer_state_embedding, seller_state_embedding] )

            # fix for sklearn regressors
            state_embedding = state_embedding.reshape((1, -1))
            q_value = self.regression_model.predict(state_embedding)[0]
            #q_value = self.regression_model.predict( state_embedding )

        if self.name == "buyer":
            q_value[0], q_value[1] = q_value[0], q_value[1]
        elif self.name == "seller":
            q_value[1], q_value[0] = q_value[0], q_value[1]
        else:
            raise ValueError("Name of mirror agent must be specified (buyer, seller) before using")

        greed_action_id = np.argmax(q_value)

        # Idea: close potentially unprofitable deals
        #max_q = np.amax( q_value )
        #if max_q < 0.0:
        #    greed_action_id = 1

        return greed_action_id