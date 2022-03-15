
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
import time
import copy

class W2VCompositeGenerator():
    def __init__(self, genList, flatStack = True):
        self.genList = genList
        self.flatStack = flatStack
        self.featureList = []
        for gen in self.genList:
            for feat in gen.featureList:
                self.featureList.append(feat)
        self.featureListSize = 0
        for gen in genList:
            self.featureListSize += len(gen.featureList)
        self.featureShape = None
        if self.flatStack:
            self.featureShape = 0
            featRowLen = 0
            for gen in genList:
                featRowLen += gen.featureShape[1]
            self.featureShape = (genList[0].featureShape[0], featRowLen)
        else:
            matrixCount = 0
            for gen in genList:
                matrixCount += gen.featureShape[0]
            self.featureShape = (matrixCount, genList[0].featureShape[1], genList[0].featureShape[2])
        pass

    def globalFit(self, df):

        for gen in self.genList:
            gen.globalFit(df)

        return self
    #propose that historyData dataframe has datetime index
    def getFeatByDatetime(self, datetimeStr, historyData):

        obs = []
        for gen in self.genList:
            genObs = np.array( gen.getFeatByDatetime(datetimeStr, historyData) )
            obs.append( genObs )

        if self.flatStack:
            obs = np.hstack(obs)
        else:
            obs = np.vstack( obs )
            obs = np.reshape(obs, [1, self.featureShape[0], self.featureShape[1], self.featureShape[2]])
            obs = np.expand_dims(obs, axis=-1)

        return obs

    def getMinDate(self, historyData):

        minDates = []
        for gen in self.genList:
            minDates.append( gen.getMinDate(historyData) )

        minDatesCopy = copy.deepcopy(minDates)
        for i in range(len(minDatesCopy)):
            minDatesCopy[i] = parser.parse(str(minDatesCopy[i])).timetuple()
            minDatesCopy[i] = time.mktime(minDatesCopy[i])
        maxDateInd = np.argmax(minDatesCopy)

        return minDates[maxDateInd]