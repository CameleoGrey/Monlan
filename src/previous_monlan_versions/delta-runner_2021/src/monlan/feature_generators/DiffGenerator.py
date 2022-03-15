
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime

class DiffGenerator():
    def __init__(self , featureList, nDiffs, nPoints = 1, flatStack = True, fitOnStep = True):
        self.scaler = StandardScaler()
        self.featureList = featureList
        self.featureListSize = len( featureList )
        self.nDiffs = nDiffs
        self.fitOnStep = fitOnStep
        self.nPoints = nPoints
        self.flatStack = flatStack
        self.featureShape = None
        if self.flatStack:
            self.featureShape = (self.nPoints * len(self.featureList),)
        else:
            self.featureShape = (self.nPoints, len(self.featureList))
        pass

    def globalFit(self, df):

        dfCopy = df.copy()
        for feat in self.featureList:
            minArr = dfCopy[feat].values
            maxArr = dfCopy[feat].values
            for i in range(self.nDiffs):
                minVal = np.min(minArr)
                maxVal = np.max(maxArr)
                minArr = minArr - minVal
                maxArr = maxArr - maxVal
            minArr = np.reshape(minArr, (-1, 1))
            maxArr = np.reshape(minArr, (-1, 1))
            self.scaler.partial_fit( minArr )
            self.scaler.partial_fit( maxArr )

        return self

    #propose that historyData dataframe has datetime index
    def getFeatByDatetime(self, datetimeStr, historyData):
        obs = None
        if self.nPoints > 1:
            obs = self.getManyPointsFeat(datetimeStr, historyData)
        else:
            obs = self.getOnePointFeat(datetimeStr, historyData)

        return obs

    def getOnePointFeat(self, datetimeStr, historyData):
        """if self.fitOnStep:
                    for feat in self.featureList:
                        tmp = obs[feat]
                        tmp = tmp.reshape(-1, 1)
                        self.scaler = self.scaler.partial_fit(tmp)"""
        df = historyData.copy()
        for i in range(self.nDiffs):
            for feat in self.featureList:
                notShifted = df[feat]
                shiftedData = df[feat].shift(periods=1)
                df[feat] = notShifted - shiftedData
            iter = next(df.iterrows())
            df = df.drop(iter[0])

        obs = df.loc[datetimeStr].copy()
        for feat in self.featureList:
            data = obs[feat]
            data = data.reshape(-1, 1)
            data = self.scaler.transform(data)
            obs[feat] = data

        selectedList = []
        for feat in self.featureList:
            selectedList.append(obs[feat])
        obs = np.array(selectedList)

        return obs

    def getManyPointsFeat(self, datetimeStr, historyData):

        df = historyData.copy()
        for i in range(self.nDiffs):
            for feat in self.featureList:
                notShifted = df[feat]
                shiftedData = df[feat].shift(periods=1)
                df[feat] = notShifted - shiftedData
            iter = next(df.iterrows())
            df = df.drop(iter[0])

        obsList = df.loc[:str(datetimeStr)].tail(self.nPoints).copy()

        if self.fitOnStep:
            for obs in obsList.iterrows():
                for feat in self.featureList:
                    tmp = obs[1][feat]
                    tmp = tmp.reshape(-1, 1)
                    self.scaler = self.scaler.partial_fit(tmp)
        obsListTransformed = {}
        for obs in obsList.iterrows():
            for feat in self.featureList:
                data = obs[1][feat]
                data = data.reshape(-1, 1)
                data = self.scaler.transform(data)
                obs[1][feat] = data
            obsListTransformed[obs[0]] = obs[1]
        obsList = pd.DataFrame(obsListTransformed.values(), index=list(obsListTransformed.keys()))

        selectedList = []
        for obs in obsList.iterrows():
            obsFeat = []
            for feat in self.featureList:
                obsFeat.append(obs[1][feat])
            selectedList.append(obsFeat)
        obs = np.array([])
        if self.flatStack:
            for i in range(len(selectedList)):
                obs = np.hstack([obs, selectedList[i]])
        else:
            obs = np.array(selectedList)

        return obs

    #get minimal date string from wich features can be generated
    def getMinDate(self, historyData):

        df = historyData.copy()
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