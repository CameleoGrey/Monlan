
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

class NoneGenerator():
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

        return None

    def get_feats_(self, history_price_array, price_feature_names_dict, history_step_id ):


        return None