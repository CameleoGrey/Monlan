
from sklearn.preprocessing import MinMaxScaler
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

class W2VScaleGenerator():
    def __init__(self , featureList, nPoints = 1, flatStack = False, fitOnStep = False,
                 nIntervals = 1000, w2vSize=50, window=10, iter=5, min_count=0, sample=0.0, sg=0):
        self.scaler = MinMaxScaler( feature_range=(0, 1) )
        self.featureList = featureList
        self.featureListSize = len( featureList )
        self.nIntervals = nIntervals
        self.intervalDict = None
        self.fitOnStep = fitOnStep
        self.nPoints = nPoints
        self.flatStack = flatStack
        self.w2vSize = w2vSize
        self.window = window
        self.iter = iter
        self.min_count = min_count
        self.sample = sample
        self.sg = sg
        self.w2vModel = None
        self.w2vDict = None
        self.featureShape = None
        if self.flatStack:
            self.featureShape = (self.nPoints, self.w2vSize * self.featureListSize)
        else:
            self.featureShape = (self.featureListSize, self.nPoints, self.w2vSize)

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
            tmp = dfCopy[feat].values
            tmp = tmp.reshape(-1, 1)
            self.scaler = self.scaler.partial_fit(tmp)

        #scale
        for feat in self.featureList:
            tmp = dfCopy[feat].values
            tmp = tmp.reshape(-1, 1)
            tmp = self.scaler.transform(tmp)
            dfCopy[feat] = tmp

        if self.fitMode:
            self.fitData = dfCopy.copy()
            self.fitData.set_index("datetime", drop=True, inplace=True)

        #build interval dict
        self.intervalDict = self.buildIntervalDict()
        #convert interval values to str
        textPrices = self.convertPricesToText(dfCopy)
        #train w2v model
        self.w2vModel, self.w2vDict = self.createW2VModelAndDict(textPrices)

        return self

    def buildIntervalDict(self):
        maxVal = self.scaler.feature_range[1]
        minVal = self.scaler.feature_range[0]
        step = (maxVal - minVal) / self.nIntervals
        intDict = {}
        for i in range(1, self.nIntervals + 1):
            intDict[minVal + i * step] = str(minVal + i * step)
        return intDict

    def createW2VModelAndDict(self, textPrices):

        docsList = []
        print("splitting docs")
        for feat in self.featureList:
            docsList.append(textPrices[feat])

        print("train word2vec model")
        w2vModel = word2vec.Word2Vec(docsList, size=self.w2vSize, window=self.window,
                                     workers=16, iter=self.iter, min_count=self.min_count, sample=self.sample, sg=self.sg)

        print("build w2v dict")
        w2vDict = dict(zip(w2vModel.wv.index2word, w2vModel.wv.syn0))

        return w2vModel, w2vDict

    def checkReconstructionQuality(self, df):

        #scale
        dfCopy = df.copy()
        for feat in self.featureList:
            tmp = dfCopy[feat].values
            tmp = tmp.reshape(-1, 1)
            tmp = self.scaler.transform(tmp)
            dfCopy[feat] = tmp

        #convert to string
        textPrices = self.convertPricesToText(dfCopy)

        #convert to vectors
        vectorPrices = {}
        for feat in self.featureList:
            vecWords = []
            doc = textPrices[feat]
            for word in doc:
                vecWords.append(self.w2vDict[word])
            vectorPrices[feat] = vecWords

        #get reconstructed prices
        reconstructedSeries = {}
        nStep = 0
        for feat in self.featureList:
            tmp = []
            for i in range(len(vectorPrices[feat])):
                approxVal = self.w2vModel.similar_by_vector(vectorPrices[feat][i], topn=1)
                approxVal = float(approxVal[0][0])
                tmp.append(approxVal)
                nStep += 1
                if nStep % 100 == 0:
                    print("prices processed: {:.3%}".format(nStep / (len(vectorPrices[feat]) * self.featureListSize)))
            reconstructedSeries[feat] = tmp

        for feat in self.featureList:
            plt.plot( [x for x in range(len(vectorPrices[feat]))], dfCopy[feat], c="b" )
            plt.plot( [x for x in range(len(vectorPrices[feat]))], reconstructedSeries[feat], c="g" )
            plt.show()

        pass

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

        obs = None
        if self.fitMode:
            obs = self.fitData.loc[datetimeStr].copy()
        else:
            obs = historyData.loc[datetimeStr].copy()
            #scale
            for feat in self.featureList:
                data = obs[feat]
                data = data.reshape(-1, 1)
                data = self.scaler.transform(data)
                obs[feat] = data

        #convert to w2v vector
        tmpObs = pd.DataFrame(obs.values.reshape(1, -1), columns=list(obs.index))
        vectorPrices = self.convertToVector(pd.DataFrame(tmpObs))
        obs = []
        if self.flatStack:
            localObs = np.zeros((self.w2vSize,))
            for feat in self.featureList:
                localObs = np.vstack( [localObs, vectorPrices[feat]] )
            obs = np.array( localObs[1:] )
        else:
            for feat in self.featureList:
                localObs = np.zeros((self.w2vSize, ))
                for i in range( len(vectorPrices[feat]) ):
                    localObs = np.vstack( [localObs, vectorPrices[feat][i]] )
                obs.append(localObs[1:])
                obs = np.array(obs)

        return obs

    def getManyPointsFeat(self, datetimeStr, historyData ):

        obsList = None
        if self.fitMode:
            obsList = self.fitData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
        else:
            obsList = historyData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
            """if self.fitOnStep:
                        for obs in obsList.iterrows():
                            for feat in self.featureList:
                                tmp = obs[1][feat]
                                tmp = tmp.reshape(-1, 1)
                                self.scaler = self.scaler.partial_fit(tmp)"""
            obsListTransformed = {}
            for obs in obsList.iterrows():
                for feat in self.featureList:
                    data = obs[1][feat]
                    data = data.reshape(-1, 1)
                    data = self.scaler.transform(data)
                    obs[1][feat] = data
                obsListTransformed[obs[0]] = obs[1]
            obsList = pd.DataFrame(obsListTransformed.values(), index=list(obsListTransformed.keys()))

        vectorPrices = self.convertToVector(obsList)
        obs = []
        if self.flatStack:
            localObs = np.zeros((self.w2vSize,))
            for feat in self.featureList:
                localObs = np.vstack([localObs, vectorPrices[feat]])
            obs = np.array(localObs[1:])
        else:
            for feat in self.featureList:
                localObs = np.zeros((self.w2vSize,))
                for i in range(len(vectorPrices[feat])):
                    localObs = np.vstack([localObs, vectorPrices[feat][i]])
                obs.append(localObs[1:])
        obs = np.array(obs)

        return obs

    def convertToVector(self, df):
        # convert to string
        dfCopy = df.copy()
        textPrices = self.convertPricesToText(dfCopy)

        # convert to vectors
        vectorPrices = {}
        for feat in self.featureList:
            vecWords = []
            doc = textPrices[feat]
            for word in doc:
                vecWords.append(self.w2vDict[word])
            vectorPrices[feat] = vecWords

        return vectorPrices

    def convertPricesToText(self, df):
        preprocSymbolData = {}
        borders = np.asarray(list(self.intervalDict.keys()))
        nStep = 0
        for feat in self.featureList:
            sample = []
            featVals = df[feat].values
            for featVal in featVals:
                idx = (np.abs(borders - featVal)).argmin()
                sample.append(self.intervalDict[borders[idx]])
                nStep += 1
            # sample = " ".join(sample)
            preprocSymbolData[feat] = sample
            if nStep % 1000 == 0:
                print("prices processed: {:.3%}".format(nStep / (df.shape[0] * self.featureListSize)))
        return preprocSymbolData

    def getMinDate(self, historyData):

        df = historyData.copy()
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