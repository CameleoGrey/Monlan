
import pandas as pd
import numpy as np
import copy
from collections import deque
from sklearn.preprocessing import MinMaxScaler

class Environment:

    def __init__(self, stSize=100):

        self.barDataPath = "../data/barData/"
        self.barData = None
        self.endInd = None
        self.barDataIter = None
        self.stSize = stSize
        self.currentInd = -1
        self.firstStep = True
        self.s_t = deque(maxlen=1000000000)

        pass

    def setBarData(self, symbolPair, period):

        dataFileName = symbolPair + "_" + period + ".csv"
        readedData = pd.read_csv( self.barDataPath + dataFileName, sep="\t" )
        self.barData = readedData
        self.endInd = self.barData.shape[0] - 1
        self.barDataIter = self.barData.iterrows()
        pass

    def step(self):
        feature="<CLOSE>"
        doneFlag = False

        if self.firstStep == True:
            while self.currentInd < self.stSize:
                st = next(self.barDataIter)[1][feature]
                self.s_t.append( st )
                self.currentInd += 1
            self.firstStep = False
        else:
            self.s_t.popleft()
            st = next(self.barDataIter)[1][feature]
            self.s_t.append( st )
            self.currentInd += 1

        if self.currentInd == self.endInd:
            doneFlag = True

        tmp = np.asarray( list( self.s_t ) )
        s_t = tmp[:self.stSize]
        y_t = tmp[self.stSize]

        return s_t, y_t, doneFlag

    def getStartInd(self):

        startInd = None
        if self.currentInd == -1:
            startInd = self.stSize
        else:
            startInd = self.currentInd

        return startInd

    def resetByIndex(self, dataIndex):

        self.currentInd = -1
        self.firstStep = True
        self.barDataIter = self.barData.iterrows()
        self.s_t = deque(maxlen=1000000000)

        self.step()
        for i in range(dataIndex - self.stSize):
            self.step()

        pass

    def getBasePrice(self):
        basePrice = None
        if (self.currentInd == -1):
            basePrice = self.barData.iloc[self.stSize]["<OPEN>"]
        else:
            basePrice = self.barData.iloc[self.currentInd + 1]["<OPEN>"]
        return basePrice