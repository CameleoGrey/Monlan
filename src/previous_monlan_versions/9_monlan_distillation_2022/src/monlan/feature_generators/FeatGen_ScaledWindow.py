
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

class FeatGen_ScaledWindow():
    def __init__(self, feature_list, n_points = 256, flat_stack = False, normalize_features = False):

        #self.normalizer = None
        self.means = None
        self.stds = None
        self.normalize_features = normalize_features

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

    def fit_normalizing(self, mod_df):

        price_feature_names_dict = {}
        price_feature_names = list(mod_df.columns)
        for i in range(len(price_feature_names)):
            price_feature_names_dict[price_feature_names[i]] = i
        history_price = mod_df.copy().values
        start_point = self.n_points

        #self.normalizer = StandardScaler()
        means = []
        stds = []
        for i in tqdm(range( start_point, len(history_price) ), desc="Fitting mean and std for the feature maps"):
            feature_map = self.get_feats_(history_price, price_feature_names_dict, i)
            feature_map = feature_map.reshape((feature_map.shape[0], feature_map.shape[1]))
            #self.normalizer.partial_fit( feature_map )
            #feature_map = self.normalizer.transform( feature_map )
            current_means = np.mean( feature_map, axis=1 )
            current_stds = np.std( feature_map, axis=1 )
            means.append( current_means )
            stds.append( current_stds )
        means = np.array( means )
        stds = np.array( stds )
        means = np.mean( means, axis=0 )
        stds = np.mean( stds, axis=0 )
        self.means = means
        self.stds = stds

        print("means: {} | stds: {}".format(self.means, self.stds))

        return self

    def get_feature_shape(self):
        return self.feature_shape

    def get_features(self, history_price_array, price_feature_names_dict, history_step_id, expand_dims=True):
        obs = self.get_feats_(history_price_array, price_feature_names_dict, history_step_id )

        if self.normalize_features:
            if self.means is None or self.stds is None:
                raise Exception("Means and stds are not fitted.")
            obs = obs.reshape( (obs.shape[0], obs.shape[1]) )
            for i in range(obs.shape[0]):
                obs[i] = obs[i] - self.means[i]
                obs[i] = obs[i] / self.stds[i]
            obs = np.reshape(obs, obs.shape + (1,))


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

        for feat in ["tick_volume", "delta", "cdv"]:
            if feat in self.feature_list:
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