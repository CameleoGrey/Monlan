
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

class FeatGen_ScaledWindow():
    def __init__(self, feature_list, n_points = 32, flat_stack = False):

        self.feature_list = feature_list
        self.feature_list_size = len( feature_list )
        self.n_points = n_points
        self.flat_stack = flat_stack
        self.feature_shape = None
        if self.flat_stack:
            self.feature_shape = (self.feature_list_size * self.n_points, )
        else:
            self.feature_shape = (self.feature_list_size, self.n_points)

        pass

    def get_feature_shape(self):
        return self.feature_shape

    def get_features(self, history_price_array, price_feature_names_dict, history_step_id, expand_dims=True):
        obs = self.get_feats_(history_price_array, price_feature_names_dict, history_step_id )

        if expand_dims:
            obs = np.expand_dims(obs, axis=0)

        return obs

    def get_feats_(self, history_price_array, price_feature_names_dict, history_step_id ):

        df = history_price_array
        #obs_list = df[:history_step_id + 1]
        #obs_list = pd.DataFrame( obs_list, columns=list(price_feature_names_dict.keys()) )
        #obs_list = obs_list[self.feature_list]
        #obs_list = obs_list.tail(self.n_points).copy()

        obs_list = df[history_step_id + 1 - self.n_points : history_step_id + 1]
        needful_feats = {}
        for feat in self.feature_list:
            needful_feats[feat] = obs_list[:, price_feature_names_dict[feat]]

        #################################
        # Scale each feature series to (0, 1) individually
        # to remove big picture properties from current window of local features
        scaler = MinMaxScaler(feature_range=(0, 1))
        for feat in ["open", "high", "low", "close"]:
            tmp = needful_feats[feat].reshape((-1, 1))
            scaler.partial_fit(tmp)
        for feat in ["open", "high", "low", "close"]:
            tmp = needful_feats[feat].reshape((-1, 1))
            tmp = scaler.transform(tmp)
            tmp = tmp.reshape((-1,))
            needful_feats[feat] = tmp

        for feat in ["cdv"]:
            tmp = needful_feats[feat].reshape((-1, 1))
            tmp = MinMaxScaler(feature_range=(0, 1)).fit_transform(tmp)
            tmp = tmp.reshape((-1,))
            needful_feats[feat] = tmp
        #################################

        #PlotRender().plot_price_cdv(obs_list)

        obs = []
        for feat in self.feature_list:
            obs.append( needful_feats[feat] )
        obs = np.vstack(obs)

        #obs = np.vstack( [needful_feats["open"], needful_feats["cdv"],
        #                  needful_feats["high"], needful_feats["cdv"],
        #                  needful_feats["low"], needful_feats["cdv"],
        #                  needful_feats["close"], needful_feats["cdv"],] )

        obs = np.reshape(obs, obs.shape + (1,))

        #1DConv features shape
        #obs = np.reshape(obs, (obs.shape[1], obs.shape[2]))
        #obs = obs.T
        ##################

        return obs