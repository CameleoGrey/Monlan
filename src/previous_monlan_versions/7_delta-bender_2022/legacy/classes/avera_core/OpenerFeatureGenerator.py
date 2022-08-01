
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from avera_core.TrendFinder import TrendFinder
import time
from tqdm import tqdm
import joblib
from joblib import delayed, Parallel
from sklearn.metrics import pairwise_distances

class OpenerFeatureGenerator():
    def __init__(self ,
                 featureList=["open", "close"],
                 nFeatRows = 1,
                 nPoints = 110,
                 nLevels=4,
                 flatStack = True,
                 fitOnStep = True):

        self.trendFinder = TrendFinder()

        #self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = StandardScaler()
        self.featureList = featureList
        self.featureListSize = len( featureList )
        self.fitOnStep = fitOnStep
        self.nFeatRows = nFeatRows
        self.nPoints = nPoints
        self.nLevels = nLevels
        self.flatStack = flatStack
        self.featureShape = None
        if self.flatStack:
            self.featureShape = ( 2*(3 * self.nLevels + 1 + self.nLevels +
                                  self.nLevels * (self.nLevels - 1)), )
        else:
            self.featureShape = ( self.nFeatRows, 2*(3 * self.nLevels + 1 + self.nLevels +
                                  self.nLevels * (self.nLevels - 1)) )

        pass

    def globalFit(self, df):

        dfDates = df.index.values

        minDate = self.getMinDate(df)
        currentDate = None
        currentDateInd = None
        for i in range(len(dfDates)):
            currentDate = dfDates[i]
            currentDateInd = i
            if currentDate == minDate:
                break

        featList = []
        trainDfLen = df[currentDate:].shape[0]
        for i in tqdm(range(trainDfLen), desc="Fitting scaler"):
            currentDate = dfDates[currentDateInd]
            feats = self.getOneRowFeat(currentDate, df, scaleFeats=False)
            featList.append( feats )
            currentDateInd += 1

        self.scaler.fit(featList)

        return self
    #propose that historyData dataframe has datetime index
    def getFeatByDatetime(self, datetimeStr, historyData):

        obs = None
        if self.nFeatRows > 1:
            obs = self.getManyRowsFeat(datetimeStr, historyData)
        else:
            obs = self.getOneRowFeat(datetimeStr, historyData)

        return obs

    def getOneRowFeat(self, datetimeStr, historyData, scaleFeats=True):

        obsList = historyData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
        #################################
        # scale open and close to (0, 1)
        dataScaler = MinMaxScaler()
        for feat in self.featureList:
            tmp = obsList[feat].values.reshape((-1, 1))
            dataScaler.partial_fit(tmp)
        for feat in self.featureList:
            tmp = obsList[feat].values.reshape((-1, 1))
            tmp = dataScaler.transform(tmp)
            tmp = tmp.reshape((-1,))
            obsList[feat] = tmp
        #################################
        levelFeats = self.trendFinder.getLevelFeatures(obsList, self.nPoints, self.nLevels)

        lastObs = obsList.values[-1]
        lastObs = pd.DataFrame(data=[lastObs], columns=obsList.columns)

        lastAvgPrice = np.average(lastObs[["open", "close"]].values)
        levelFeats = levelFeats + [lastAvgPrice]

        levelPriceDeltas = []
        for i in range(self.nLevels):
            levelPriceDeltas.append(levelFeats[i] - lastAvgPrice)
        levelFeats = levelFeats + levelPriceDeltas

        crossLevelDistances = []
        for i in range(self.nLevels):
            for j in range(self.nLevels):
                if i != j:
                    cld = np.sqrt(np.square(levelFeats[i] - levelFeats[j]))
                    crossLevelDistances.append(cld)
        levelFeats = levelFeats + crossLevelDistances

        ###############################
        volumeFeats = self.getOneRowVolumeFeat(datetimeStr, historyData, scaleFeats=True)
        levelFeats = levelFeats + volumeFeats
        ###############################

        if scaleFeats:
            levelFeats = np.array(levelFeats).reshape((1, -1))
            levelFeats = self.scaler.transform(levelFeats)
            levelFeats = levelFeats.reshape((-1,))

        obs = levelFeats
        return obs

    def getOneRowVolumeFeat(self, datetimeStr, historyData, scaleFeats=True):

        obsList = historyData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
        #################################
        # scale open and close to (0, 1)
        dataScaler = MinMaxScaler()
        for feat in ["tick_volume"]:
            tmp = obsList[feat].values.reshape((-1, 1))
            dataScaler.partial_fit(tmp)
        for feat in ["tick_volume"]:
            tmp = obsList[feat].values.reshape((-1, 1))
            tmp = dataScaler.transform(tmp)
            tmp = tmp.reshape((-1,))
            obsList[feat] = tmp
        #################################
        levelFeats = self.trendFinder.getVolumeFeatures(obsList, self.nPoints, self.nLevels)

        lastObs = obsList.values[-1]
        lastObs = pd.DataFrame(data=[lastObs], columns=obsList.columns)

        lastAvgPrice = np.average(lastObs[["tick_volume"]].values)
        levelFeats = levelFeats + [lastAvgPrice]

        levelPriceDeltas = []
        for i in range(self.nLevels):
            levelPriceDeltas.append(levelFeats[i] - lastAvgPrice)
        levelFeats = levelFeats + levelPriceDeltas

        crossLevelDistances = []
        for i in range(self.nLevels):
            for j in range(self.nLevels):
                if i != j:
                    cld = np.sqrt(np.square(levelFeats[i] - levelFeats[j]))
                    crossLevelDistances.append(cld)
        levelFeats = levelFeats + crossLevelDistances

        obs = levelFeats
        return obs

    def getManyRowsFeat(self, datetimeStr, historyData):

        def processBatch(obsBatch):
            processedObs = []
            for ob in obsBatch:
                lastDt = ob.index.values[-1]
                tmp = self.getOneRowFeat(lastDt, ob, scaleFeats=True)
                processedObs.append(tmp)
            return processedObs

        lastData = historyData.loc[:str(datetimeStr)].tail(self.nPoints + self.nFeatRows - 1)
        lastData.reset_index(inplace=True)

        obsLists = []
        for i in range(self.nFeatRows):
            obsList = lastData[i:i + self.nPoints]
            obsLists.append(obsList)
        for i in range(len(obsLists)):
            obsLists[i].set_index("datetime", drop=True, inplace=True)

        n_jobs = 4
        batchSize = self.nFeatRows // n_jobs
        batches = []
        for i in range(batchSize - 1):
            batches.append(obsLists[i * batchSize: (i + 1) * batchSize])
        batches.append(obsLists[(batchSize - 1) * batchSize:])

        obsLists = Parallel(n_jobs=n_jobs)(
            delayed(processBatch)(batch) for batch in batches)
        obs = np.vstack(obsLists)
        return obs

    def getAdaptiveLimits(self, datetimeStr, historyData, dealOpenPrice, spreadCoef, stopPos = 0, takePos = 0):

        obsList = historyData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
        avgVals, levelLabels = self.trendFinder.findLevels(obsList, self.nPoints, self.nLevels)
        uniqLevels = np.sort(np.unique(levelLabels))
        avgLevels = np.zeros((uniqLevels.shape[0],))
        for i in range(uniqLevels.shape[0]):
            avgLevels[i] = np.average( avgVals[ levelLabels == uniqLevels[i] ] )
        dists = pairwise_distances(np.array(dealOpenPrice).reshape((-1, 1)), avgLevels.reshape((-1, 1)), metric="euclidean", n_jobs=1)
        dists = dists[0]

        sortedDists = np.sort(dists)
        limitsDict = {}
        limitsDict["stop"] = int(sortedDists[stopPos] / spreadCoef)
        limitsDict["take"] = int(sortedDists[takePos] / spreadCoef)

        return limitsDict

    def getMinDate(self, historyData):

        df = historyData.copy()
        iter = df.iterrows()
        obs = next(iter)
        for i in range(self.nPoints + self.nFeatRows - 1):
            obs = next(iter)

        return obs[0]