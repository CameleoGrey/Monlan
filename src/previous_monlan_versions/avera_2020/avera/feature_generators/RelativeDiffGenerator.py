
from sklearn.preprocessing import StandardScaler
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

class RelativeDiffGenerator():
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

        for i in range(self.nDiffs):
            for feat in self.featureList:
                notShifted = dfCopy[feat]
                shiftedData = dfCopy[feat].shift(periods=1)
                dfCopy[feat] = notShifted - shiftedData
            iter = next(dfCopy.iterrows())
            dfCopy = dfCopy.drop(iter[0])

        dfCopy = df.copy()
        for feat in self.featureList:
            minArr = dfCopy[feat].values
            maxArr = dfCopy[feat].values
            minVal = np.min(minArr)
            maxVal = np.max(maxArr)
            minArr = minArr - minVal
            maxArr = maxArr - maxVal
            minArr = np.reshape(minArr, (-1, 1))
            maxArr = np.reshape(maxArr, (-1, 1))
            self.scalerDict[feat].partial_fit( minArr )
            self.scalerDict[feat].partial_fit( maxArr )

        if self.fitMode:
            self.fitData = dfCopy.copy()
            self.fitData.set_index("datetime", drop=True, inplace=True)

        return self

    #propose that historyData dataframe has datetime index
    def getFeatByDatetime(self, datetimeStr, historyData, expandDims=True):
        obs = self.getManyPointsFeat(datetimeStr, historyData)

        if expandDims:
            obs = np.expand_dims(obs, axis=0)

        return obs

    def getManyPointsFeat(self, datetimeStr, historyData ):

        obsList = None
        if self.fitMode:
            obsList = self.fitData.loc[:str(datetimeStr)].tail(self.nPoints + 1).copy()
            lastPoint = obsList.tail(1)
            obsList = obsList.head(self.nPoints)
            for feat in self.featureList:
                featVals = obsList[feat].values
                featVals = featVals - lastPoint[feat].values[0]
                featVals = np.reshape(featVals, (-1, 1))
                featVals = self.scalerDict[feat].transform(featVals)
                obsList[feat] = featVals
        else:
            df = historyData.copy()

            ########
            df = df[:str(datetimeStr)].tail(500)
            from avera.mods.VSASpread import VSASpread
            from avera.mods.HeikenAshiMod import HeikenAshiMod
            from avera.mods.EnergyMod import EnergyMod
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

        obs = np.reshape(obs, (obs.shape[1], obs.shape[2], 1))

        return obs

    def getMinDate(self, historyData):

        df = historyData.copy()

        ########
        """from avera.mods.VSASpread import VSASpread
        from avera.mods.HeikenAshiMod import HeikenAshiMod
        from avera.mods.EnergyMod import EnergyMod
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
        for i in range(self.nPoints):
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