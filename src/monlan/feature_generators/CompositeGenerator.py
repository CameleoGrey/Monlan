
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
import time
import copy
import joblib

class CompositeGenerator():
    def __init__(self, gen_list=[], flat_stack = False):
        self.gen_list = gen_list
        self.flat_stack = flat_stack
        self.feature_list = []

        if len(gen_list) != 0:
            for gen in self.gen_list:
                for feat in gen.feature_list:
                    self.feature_list.append(feat)
            self.feature_listSize = 0
            for gen in gen_list:
                self.feature_listSize += len(gen.feature_list)
            self.feature_shape = None
            if self.flat_stack:
                self.feature_shape = 0
                for gen in gen_list:
                    self.feature_shape += gen.feature_shape[0]
                self.feature_shape = (self.feature_shape,)
            else:
                cols = gen_list[0].feature_shape[1]
                rows = 0
                for gen in gen_list:
                    rows += gen.feature_shape[0]
                self.feature_shape = (rows, cols)

        pass

    def global_fit(self, df):

        for gen in self.gen_list:
            gen.global_fit(df)

        return self

    def get_feat_by_datetime(self, datetimeStr, history_data):

        obs = []
        for gen in self.gen_list:
            gen_obs = np.array(gen.get_feat_by_datetime(datetimeStr, history_data, expandDims=False))
            obs.append(gen_obs)

        if self.flat_stack:
            obs = np.hstack(obs)
        else:
            obs = np.vstack(obs)

        obs = np.expand_dims(obs, axis=0)

        return obs

    def set_fit_mode(self, fit_mode):
        for gen in self.gen_list:
            gen.set_fit_mode(fit_mode)
        pass

    def get_min_date(self, history_data):

        min_dates = []
        for gen in self.gen_list:
            min_dates.append( gen.get_min_date(history_data) )

        min_dates_copy = copy.deepcopy(min_dates)
        for i in range(len(min_dates_copy)):
            min_dates_copy[i] = parser.parse(str(min_dates_copy[i])).timetuple()
            min_dates_copy[i] = time.mktime(min_dates_copy[i])
        max_date_ind = np.argmax(min_dates_copy)

        return min_dates[max_date_ind]

    def save_generator(self, path):
        with open(path, 'wb') as genFile:
            joblib.dump(self, genFile)
        pass

    def load_generator(self, path):
        generator = None
        with open(path, "rb") as genFile:
            generator = joblib.load(genFile)
        return generator