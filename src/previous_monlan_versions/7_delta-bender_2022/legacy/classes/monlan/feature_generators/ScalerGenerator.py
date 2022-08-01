
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
import time

class ScalerGenerator():
    def __init__(self , featureList, nPoints = 1, flatStack = True, fitOnStep = True):
        self.scaler = StandardScaler()
        self.featureList = featureList
        self.featureListSize = len( featureList )
        self.fitOnStep = fitOnStep
        self.nPoints = nPoints
        self.flatStack = flatStack
        self.featureShape = None
        if self.flatStack:
            self.featureShape = (self.nPoints * len(self.featureList),)
        else:
            self.featureShape = ( self.nPoints, len(self.featureList) )

        pass

    def globalFit(self, df):

        dfCols = list(df.columns)
        for feat in self.featureList:
            if feat not in dfCols:
                raise ValueError("df doesn't contain column: \"{}\" ".format(feat))

        # if len( featList ) == 0:
        #    featList = list( df.columns )
        #    featList.remove("datetime")

        for feat in self.featureList:
            tmp = df[feat].values
            tmp = tmp.reshape(-1, 1)
            self.scaler = self.scaler.partial_fit(tmp)

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

        obs = historyData.loc[datetimeStr].copy()
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

    def getManyPointsFeat(self, datetimeStr, historyData ):
        """dateTail = list(historyData.tail(2).index)
        prevDate = parser.parse(str(dateTail[0]))
        lastDate = parser.parse(str(dateTail[1]))
        dateDelta = lastDate - prevDate

        startDate = parser.parse(str(datetimeStr))
        endDate = parser.parse(str(datetimeStr))
        for i in range(self.nPoints - 1):
            startDate = startDate - dateDelta"""

        #obsList = historyData.loc[str(startDate): str(datetimeStr)].copy()
        obsList = historyData.loc[:str(datetimeStr)].tail(self.nPoints).copy()

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

    def getMinDate(self, historyData):

        df = historyData.copy()
        iter = df.iterrows()
        obs = next(iter)
        for i in range(self.nPoints - 1):
            obs = next(iter)


        return obs[0]