import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

import os

from scipy.spatial.distance import sqeuclidean
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataManager:

    def __init__(self):

        self.barDataPath = "./data/barData/"

        self.barDataScalers = []
        self.indOfTargetSeq = -1

        pass

    def readBarData(self, symbolPair, period):

        dataFileName = symbolPair + "_" + period + ".csv"

        if os.path.exists( self.barDataPath + dataFileName ):
            readedData = pd.read_csv( self.barDataPath + dataFileName, sep="\t" )

        else:
            print ("Bar data doesn't exist: " + self.barDataPath + dataFileName)
            exit(666)

        return readedData

    def onlineBarDataUpdate(self, symbolPair, period):



        pass

    def updateBarData(self, updateDataFilePath, symbolPair, period, fullUpdate=True):

        updateFile = open(updateDataFilePath, "r")

        oldDataPath = self.barDataPath + symbolPair + "_" + period + ".csv"
        oldFile = None
        if (fullUpdate == True):
            oldFile = open(oldDataPath, "w")
        else:
            oldFile = open(oldDataPath, "w+")

        for line in updateFile:
            oldFile.write(line)

        updateFile.close()
        oldFile.close()

        pass

    def getSetToPredict(self,
                        barData,
                        domainColumns = ["<OPEN>", "<CLOSE>", "<LOW>", "<HIGH>"],
                        modelIOParams = {"inputSeqCount" : 4, "inputSeqLength" : 200, "predictSeqLength" : 12}
                        ):

        N = modelIOParams["inputSeqCount"]
        L = modelIOParams["inputSeqLength"]
        P = modelIOParams["predictSeqLength"]

        domainBarData = barData.tail(2 * L)
        domainBarData = domainBarData[domainColumns].values

        dbDim = domainBarData.shape[0]
        domainSet = np.zeros((N, dbDim - L + 1, L))

        for i in range(domainSet.shape[0]):
            for j in range(domainSet.shape[1]):
                for k in range(domainSet.shape[2]):
                    domainSet[i][j][k] = domainBarData[j + k][i]


        return domainSet


    def makeDataSetsFromBarData(self,
                                barData,
                                splitCoef = 0.1,
                                domainColumns = ["<OPEN>", "<CLOSE>", "<LOW>", "<HIGH>"],
                                targetSeq = ["<CLOSE>"],
                                modelIOParams = {"inputSeqCount" : 4, "inputSeqLength" : 200, "predictSeqLength" : 12}
                                ):

        self.barDataScalers = self.getScalersForBarData(barData, domainColumns)
        self.indOfTargetSeq = self.getIndexOfTargetColumn(barData, domainColumns, targetSeq)

        barData = barData[domainColumns]

        targetSeq = barData[targetSeq].values
        targetSeq = np.reshape(targetSeq, newshape=(-1, 1))

        for colName in barData.keys():
            colVals = barData[colName].values
            colVals = np.reshape(colVals, newshape=(-1, 1))
            barData[colName] = colVals
        barData = barData.values

        N = modelIOParams["inputSeqCount"]
        L = modelIOParams["inputSeqLength"]
        P = modelIOParams["predictSeqLength"]

        samplesCount = barData.shape[0]
        trainSamplesCount = int( samplesCount * splitCoef )
        testSamplesCount = int( samplesCount - trainSamplesCount ) - L - P + 1

        X_train = np.zeros((N, trainSamplesCount, L))
        Y_train = np.zeros((trainSamplesCount, P))

        X_test = np.zeros((N, testSamplesCount, L))
        Y_test = np.zeros((testSamplesCount, P))

        for i in range(N):
            for j in range(trainSamplesCount):
                for k in range(L):
                    X_train[i][j][k] = barData[j + k][i]

        for i in range(0, trainSamplesCount):
            for j in range(0, P):
                Y_train[i][j] = targetSeq[L + 1 + i + j]

        for i in range(N):
            for j in range(testSamplesCount):
                for k in range(L):
                    X_test[i][j][k] = barData[trainSamplesCount + j + k][i]

        for i in range(0, testSamplesCount):
            for j in range(0, P):
                Y_test[i][j] = targetSeq[trainSamplesCount + L + i + j]

        return (X_train, Y_train), (X_test, Y_test)

    def scaleXYDataSets(self, X_dataSet, Y_dataSet, isYsubSequenceOfX=True):

        N = X_dataSet.shape[0]
        K = X_dataSet.shape[1]
        L = X_dataSet.shape[2]

        xSeq = np.zeros((K + L - 1, ))

        """for i in range(N):
            for k in range(K):
                xSeq[k] = X_dataSet[i][k][0]

            for k in range(L):
                xSeq[K + k] = X_dataSet[i][K-1][k]"""

        for i in range(N):
            for j in range(K):
                for k in range(L):
                    xSeq[j + k] = X_dataSet[i][j][k]

            xSeq = np.reshape(xSeq, newshape=(-1, 1))
            xSeq = self.barDataScalers[i].transform(xSeq)

            for k in range(K):
                for j in range(L):
                    X_dataSet[i][k][j] = xSeq[k + j]

        Y_dataSet = self.barDataScalers[self.indOfTargetSeq].transform(Y_dataSet)

        return (X_dataSet, Y_dataSet)

    def scalePredictDataSet(self, X_dataSet):

        N = X_dataSet.shape[0]
        K = X_dataSet.shape[1]
        L = X_dataSet.shape[2]

        xSeq = np.zeros((K + L - 1, ))

        for i in range(N):
            for j in range(K):
                for k in range(L):
                    xSeq[j + k] = X_dataSet[i][j][k]

            #for k in range(K - L - 1, K):
            #xSeq[k] = X_dataSet[i][K-1][k]

            xSeq = np.reshape(xSeq, newshape=(-1, 1))
            xSeq = self.barDataScalers[i].transform(xSeq)

            for k in range(K):
                for j in range(L):
                    X_dataSet[i][k][j] = xSeq[k + j]

        return X_dataSet

    def getScalersForBarData(self, barData, domainColumns=["<OPEN>", "<CLOSE>", "<LOW>", "<HIGH>"]):

        targetBarData = barData[domainColumns]
        scalers = []

        for colName in targetBarData.keys():
            colVals = barData[colName].values
            colVals = np.reshape(colVals, newshape=(-1, 1))
            scaler  = MinMaxScaler((-1, 1), False)
            scaler.fit(colVals)
            scalers.append( copy.deepcopy(scaler) )

        return scalers

    def getIndexOfTargetColumn(self, barData, domainColumns, targetColumn):

        barData = barData[domainColumns]
        ind = barData.columns.get_loc(targetColumn[0])

        return ind
