
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

class IdentFeatGen():
    def __init__(self, featureList, nPoints = 32, flatStack = True):

        self.featureList = featureList
        self.featureListSize = len( featureList )
        self.intervalDict = None
        self.nPoints = nPoints
        self.flatStack = flatStack
        self.iter = iter
        self.featureShape = None
        if self.flatStack:
            self.featureShape = (self.featureListSize * self.nPoints, )
        else:
            self.featureShape = (self.featureListSize, self.nPoints)

        self.fitMode = False
        self.fitData = None

        pass

    def get_feature_shape(self):
        return self.featureShape

    #propose that historyData dataframe has datetime index
    def get_features(self, history_price_array, price_feature_names_dict, history_step_id, expandDims=False):
        obs = self.get_feats_(history_price_array, price_feature_names_dict, history_step_id )

        if expandDims:
            obs = np.expand_dims(obs, axis=0)

        return obs

    def get_feats_(self, history_price_array, price_feature_names_dict, history_step_id ):

        df = history_price_array
        #obsList = df[:history_step_id + 1]
        #obsList = pd.DataFrame( obsList, columns=list(price_feature_names_dict.keys()) )
        #obsList = obsList[self.featureList]
        #obsList = obsList.tail(self.nPoints).copy()

        obsList = df[history_step_id + 1 - self.nPoints : history_step_id + 1]
        needful_feats = {}
        for feat in self.featureList:
            needful_feats[feat] = obsList[:, price_feature_names_dict[feat]]

        obs = []
        for feat in self.featureList:
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