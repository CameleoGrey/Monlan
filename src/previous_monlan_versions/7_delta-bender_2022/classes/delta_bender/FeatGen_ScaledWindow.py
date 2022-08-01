
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from  classes.delta_bender.PlotRender import PlotRender

class FeatGen_ScaledWindow():
    def __init__(self , featureList, nPoints = 64, nDiffs=0, flatStack = False):

        self.featureList = featureList
        self.featureListSize = len( featureList )
        self.intervalDict = None
        self.nPoints = nPoints
        self.nDiffs = nDiffs
        self.flatStack = flatStack
        self.iter = iter
        self.featureShape = None
        if self.flatStack:
            self.featureShape = (self.featureListSize * self.nPoints, )
        else:
            #self.featureShape = (self.featureListSize, self.nPoints)
            self.featureShape = (self.featureListSize, self.nPoints)

        self.fitMode = False
        self.fitData = None

        pass

    #propose that historyData dataframe has datetime index
    def get_window(self, ind, historyData, expandDims=True):
        obs = self.getManyPointsFeat(ind, historyData)

        if expandDims:
            obs = np.expand_dims(obs, axis=0)

        return obs

    def getManyPointsFeat(self, ind, historyData ):
        df = historyData.copy()

        for i in range(self.nDiffs):
            for feat in self.featureList:
                notShifted = df[feat]
                shiftedData = df[feat].shift(periods=1)
                df[feat] = notShifted - shiftedData
            iter = next(df.iterrows())
            df = df.drop(iter[0])

        obsList = df.loc[:ind]
        obsList = obsList.tail(self.nPoints).copy()

        #################################
        # Scale each feature series to (0, 1) individually
        # to remove big picture properties from current window of local features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for feat in ["open", "high", "low", "close"]:
            tmp = obsList[feat].values.reshape((-1, 1))
            scaler.partial_fit(tmp)
        for feat in ["open", "high", "low", "close"]:
            tmp = obsList[feat].values.reshape((-1, 1))
            tmp = scaler.transform(tmp)
            tmp = tmp.reshape((-1,))
            obsList[feat] = tmp

        for feat in ["cdv"]:
            tmp = obsList[feat].values.reshape((-1, 1))
            tmp = MinMaxScaler(feature_range=(-1, 1)).fit_transform(tmp)
            tmp = tmp.reshape((-1,))
            obsList[feat] = tmp
        #################################

        #PlotRender().plot_price_cdv(obsList)

        obs = obsList.values
        #obs = np.vstack( [obsList["open"].values, obsList["cdv"].values,
        #                  obsList["high"].values, obsList["cdv"].values,
        #                  obsList["low"].values, obsList["cdv"].values,
        #                  obsList["close"].values, obsList["cdv"].values,] )
        obs = np.reshape(obs, obs.shape + (1,))

        #1DConv features shape
        #obs = np.reshape(obs, (obs.shape[1], obs.shape[2]))
        #obs = obs.T
        ##################

        return obs