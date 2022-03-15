
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
import time
import copy
import joblib

class CompositeGenerator():
    def __init__(self, genList=[], flatStack = False):
        self.genList = genList
        self.flatStack = flatStack
        self.featureList = []

        if len(genList) != 0:
            for gen in self.genList:
                for feat in gen.featureList:
                    self.featureList.append(feat)
            self.featureListSize = 0
            for gen in genList:
                self.featureListSize += len(gen.featureList)
            self.featureShape = None
            if self.flatStack:
                self.featureShape = 0
                for gen in genList:
                    self.featureShape += gen.featureShape[0]
                self.featureShape = (self.featureShape,)
            else:
                cols = genList[0].featureShape[1]
                rows = 0
                for gen in genList:
                    rows += gen.featureShape[0]
                self.featureShape = (rows, cols)

        pass

    def globalFit(self, df):

        for gen in self.genList:
            gen.globalFit(df)

        return self
    #propose that historyData dataframe has datetime index
    def getFeatByDatetime(self, datetimeStr, historyData):

        obs = []
        for gen in self.genList:
            genObs = np.array(gen.getFeatByDatetime(datetimeStr, historyData, expandDims=False))
            obs.append(genObs)

        if self.flatStack:
            obs = np.hstack(obs)
        else:
            obs = np.vstack(obs)

        obs = np.expand_dims(obs, axis=0)

        return obs

    def setFitMode(self, fitMode):
        for gen in self.genList:
            gen.setFitMode(fitMode)
        pass

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

    def saveGenerator(self, path):
        with open(path, 'wb') as genFile:
            joblib.dump(self, genFile)
        pass

    def loadGenerator(self, path):
        generator = None
        with open(path, "rb") as genFile:
            generator = joblib.load(genFile)
        return generator