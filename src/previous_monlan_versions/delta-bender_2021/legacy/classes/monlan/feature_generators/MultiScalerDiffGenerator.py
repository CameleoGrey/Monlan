
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
import time
import joblib
import gensim
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import matplotlib.pyplot as plt

class MultiScalerDiffGenerator():
    def __init__(self , featureList, nPoints = 1, nDiffs=1, flatStack = False, fitOnStep = False):

        self.scalerDict = {}
        for feat in featureList:
            self.scalerDict[feat] = StandardScaler()
        self.featureList = featureList
        self.featureListSize = len( featureList )
        self.intervalDict = None
        self.fitOnStep = fitOnStep
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

    def setFitMode(self, fitMode):
        if fitMode == True:
            self.fitMode = True
        else:
            self.fitMode = False
            self.fitData = None
        pass

    def globalFit(self, df):

        dfCopy = df.copy()

        dfCols = list(dfCopy.columns)
        for feat in self.featureList:
            if feat not in dfCols:
                raise ValueError("df doesn't contain column: \"{}\" ".format(feat))

        for feat in self.featureList:
            diffVals = dfCopy.copy()
            for i in range(self.nDiffs):
                notShifted = diffVals[feat]
                shiftedData = diffVals[feat].shift(periods=1)
                diffVals[feat] = notShifted - shiftedData
            tmp = diffVals[feat].values
            tmp = np.reshape(tmp, (-1, 1))
            self.scalerDict[feat].fit( tmp )

        for i in range(self.nDiffs):
            for feat in self.featureList:
                notShifted = dfCopy[feat]
                shiftedData = dfCopy[feat].shift(periods=1)
                dfCopy[feat] = notShifted - shiftedData
            iter = next(dfCopy.iterrows())
            dfCopy = dfCopy.drop(iter[0])

        for feat in self.featureList:
            data = dfCopy[feat].values
            data = data.reshape(-1, 1)
            data = self.scalerDict[feat].transform(data)
            dfCopy[feat] = data

        if self.fitMode:
            self.fitData = dfCopy.copy()
            self.fitData.set_index("datetime", drop=True, inplace=True)

        return self

    #propose that historyData dataframe has datetime index
    def getFeatByDatetime(self, datetimeStr, historyData, expandDims=True):
        obs = self.getManyPointsFeat(datetimeStr, historyData)

        if expandDims:
            obs = np.expand_dims(obs, axis=0)

        ###########################
        #tmp = []
        #for i in range(3):
        #    tmp.append(obs)
        #obs = np.hstack(tmp)
        ##########################

        return obs

    def getManyPointsFeat(self, datetimeStr, historyData ):

        obsList = None
        if self.fitMode:
            obsList = self.fitData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
        else:
            df = historyData.copy()

            ########
            df = df[:str(datetimeStr)].tail(500)
            from monlan.mods.VSASpread import VSASpread
            from monlan.mods.HeikenAshiMod import HeikenAshiMod
            from monlan.mods.EnergyMod import EnergyMod
            historyMod = VSASpread()
            df = historyMod.modHistory(df)
            historyMod = HeikenAshiMod()
            df = historyMod.modHistory(df)
            historyMod = EnergyMod()
            df = historyMod.modHistory(df, featList=["open", "close", "low", "high"])
            ########

            for i in range(self.nDiffs):
                for feat in self.featureList:
                    notShifted = df[feat]
                    shiftedData = df[feat].shift(periods=1)
                    df[feat] = notShifted - shiftedData
                iter = next(df.iterrows())
                df = df.drop(iter[0])

            obsList = df.loc[:str(datetimeStr)].tail(self.nPoints).copy()

            obsListTransformed = {}
            for obs in obsList.iterrows():
                for feat in self.featureList:
                    data = obs[1][feat]
                    data = data.reshape(-1, 1)
                    data = self.scalerDict[feat].transform(data)
                    obs[1][feat] = data
                obsListTransformed[obs[0]] = obs[1]
            obsList = pd.DataFrame(obsListTransformed.values(), index=list(obsListTransformed.keys()))

        #################################
        # Scale each feature series to (0, 1) individually
        # to remove big picture properties from current window of local features
        for feat in self.featureList:
            tmp = obsList[feat].values.reshape((-1, 1))
            tmp = MinMaxScaler().fit_transform(tmp)
            tmp = tmp.reshape((-1,))
            obsList[feat] = tmp
        #################################

        obs = []
        if self.flatStack:
            localObs = np.zeros((self.nPoints,))
            #for feat in self.featureList:
            #    localObs = np.vstack([localObs, vectorPrices[feat]])
            #obs = np.array(localObs[1:])
        else:
            localObs = np.zeros((self.nPoints,))
            for feat in self.featureList:
                localObs = np.vstack([localObs, obsList[feat].values])
            obs.append(localObs[1:])
        obs = np.array(obs)

        #2DConv features shape
        obs = np.reshape(obs, (obs.shape[1], obs.shape[2], 1))
        #1DConv features shape
        #obs = np.reshape(obs, (obs.shape[1], obs.shape[2]))
        #obs = obs.T
        ##################

        return obs

    def getMinDate(self, historyData):

        df = historyData.copy()

        ########
        """from monlan.mods.VSASpread import VSASpread
        from monlan.mods.HeikenAshiMod import HeikenAshiMod
        from monlan.mods.EnergyMod import EnergyMod
        historyMod = VSASpread()
        df = historyMod.modHistory(df)
        historyMod = HeikenAshiMod()
        df = historyMod.modHistory(df)
        historyMod = EnergyMod()
        df = historyMod.modHistory(df, featList=["open", "close", "low", "high"])"""
        ########

        for i in range(self.nDiffs):
            for feat in self.featureList:
                notShifted = df[feat]
                shiftedData = df[feat].shift(periods=1)
                df[feat] = notShifted - shiftedData
            iter = next(df.iterrows())
            df = df.drop(iter[0])
        iter = df.iterrows()
        obs = next(iter)
        for i in range(self.nPoints - 1):
            obs = next(iter)

        return obs[0]

    def saveGenerator(self, path):
        with open(path, 'wb') as genFile:
            joblib.dump(self, genFile)
        pass

    def loadGenerator(self, path):
        generator = None
        with open(path, "rb") as genFile:
            generator = joblib.load(genFile)
        return generator