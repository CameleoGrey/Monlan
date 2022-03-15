
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

class W2VRelativeDiffGenerator():
    def __init__(self , featureList, nPoints = 1, nDiffs=1, flatStack = False, fitOnStep = False,
                 nIntervals = 1000, w2vSize=50, window=10, iter=5, min_count=0, sample=0.0, sg=0):
        self.scaler = MinMaxScaler( feature_range=(0, 1) )
        self.featureList = featureList
        self.featureListSize = len( featureList )
        self.nIntervals = nIntervals
        self.intervalDict = None
        self.fitOnStep = fitOnStep
        self.nPoints = nPoints
        self.nDiffs = nDiffs
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
        pass

    def globalFit(self, df):

        dfCopy = df.copy()

        dfCols = list(dfCopy.columns)
        for feat in self.featureList:
            if feat not in dfCols:
                raise ValueError("df doesn't contain column: \"{}\" ".format(feat))
        print("global fit: fit scaler...")
        for feat in self.featureList:
            featArr = dfCopy[feat].values
            valArr = dfCopy[feat].values
            for i in range(valArr.shape[0]):
                diffArr = featArr - valArr[i]
                diffArr = np.reshape(diffArr, (-1, 1))
                self.scaler.partial_fit( diffArr )
                if i % (valArr.shape[0] // 4) == 0:
                    print("fit scaler at {}: {:.2%}".format(feat, i / valArr.shape[0]))

        print("Build data frame for training w2v")
        futureDfDict = {}
        for feat in self.featureList:
            featArr = dfCopy[feat].values
            valArr = dfCopy[feat].values
            diffList = []
            for i in range(valArr.shape[0]):
                diffArr = featArr - valArr[i]
                diffArr = np.reshape(diffArr, (-1, 1))
                diffArr = self.scaler.transform(diffArr)
                diffArr = np.reshape(diffArr, (-1, ))
                diffList.append( diffArr )
                if i % (valArr.shape[0] // 4) == 0:
                    print("build series for {}: {:.2%}".format(feat, i / valArr.shape[0]))
            futureDfDict[feat] = np.hstack(diffList)
        dfCopy = pd.DataFrame( futureDfDict )

        #build interval dict
        self.intervalDict = self.buildIntervalDict()
        #convert interval values to str
        print("Converting prices to text")
        textPrices = self.convertPricesToText(dfCopy, nJobs=16)
        #train w2v model
        print("Building w2v model")
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

    def convertPricesToText(self, df, nJobs=1):
        preprocSymbolData = {}
        borders = np.asarray(list(self.intervalDict.keys()))
        def convertChunk(chunk, borders, intervalDict):
            nStep = 0
            sample = []
            for featVal in chunk:
                idx = (np.abs( borders - featVal )).argmin()
                sample.append( intervalDict[borders[idx]] )
                nStep += 1
                if nStep % ( (chunk.shape[0] + 1) // 4) == 0:
                    print("Converting chunk: {:.2%}".format(nStep / (chunk.shape[0])))
            return sample

        from joblib import Parallel, delayed
        from copy import deepcopy
        nStep = 0
        for feat in self.featureList:
            sample = []
            featVals = df[feat].values

            if nJobs > 1:
                chunkList = []
                chunkSize = len(featVals) // nJobs
                for i in range(nJobs - 1):
                    chunkList.append(featVals[i*chunkSize : (i+1)*chunkSize])
                chunkList.append(featVals[(nJobs - 1) * chunkSize:])
                chunks = Parallel(n_jobs=nJobs)(delayed(convertChunk)(chunk, deepcopy(borders), deepcopy(self.intervalDict)) for chunk in chunkList)
                for chunk in chunks:
                    sample = sample + chunk
            else:
                for featVal in featVals:
                    idx = (np.abs( borders - featVal )).argmin()
                    sample.append( self.intervalDict[borders[idx]] )
                    nStep += 1
                    #if nStep % (df.shape[0] * self.featureListSize // 100) == 0:
                    #    print("Converting prices to text: {:.2%}".format(nStep / (df.shape[0] * self.featureListSize)))
                #sample = " ".join(sample)
            preprocSymbolData[feat] = sample
        return preprocSymbolData

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
        futureDfDict = {}
        for feat in self.featureList:
            featArr = dfCopy[feat].values
            valArr = dfCopy[feat].values
            diffList = []
            for i in range(valArr.shape[0]):
                diffArr = featArr - valArr[i]
                diffArr = np.reshape(diffArr, (-1, 1))
                diffArr = self.scaler.transform(diffArr)
                diffArr = np.reshape(diffArr, (-1,))
                diffList.append(diffArr)
            futureDfDict[feat] = np.hstack(diffList)
        dfCopy = pd.DataFrame(futureDfDict)

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

        return obs

    def getManyPointsFeat(self, datetimeStr, historyData ):
        dfCopy = historyData.copy()
        dfCopy = dfCopy.loc[:str(datetimeStr)].tail(self.nPoints + 1).copy()
        obsList = dfCopy.head(self.nPoints).copy()
        currentPoint = dfCopy.tail(1).copy()

        futureDfDict = {}
        for feat in self.featureList:
            featArr = obsList[feat].values
            valArr = currentPoint[feat].values[0]
            diffArr = featArr - valArr
            diffArr = np.reshape(diffArr, (-1, 1))
            diffArr = self.scaler.transform(diffArr)
            diffArr = np.reshape(diffArr, (-1,))
            futureDfDict[feat] = diffArr
        obsList = pd.DataFrame(futureDfDict)

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

    def saveGenerator(self, path):
        with open(path, 'wb') as genFile:
            joblib.dump(self, genFile)
        pass

    def loadGenerator(self, path):
        generator = None
        with open(path, "rb") as genFile:
            generator = joblib.load(genFile)
        return generator