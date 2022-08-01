
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from avera_core.TrendFinder import TrendFinder
import time
from tqdm import tqdm
import random
import joblib
from joblib import delayed, Parallel
from multiprocessing.dummy import Pool as ThreadPool
from avera.utils.save_load import *

import multiprocessing as mp
import os

class CloserFeatureGenerator():
    def __init__(self ,
                 featureList=["open", "close"],
                 nFeatRows = 1,
                 nPoints = 110,
                 nLevels=4,
                 flatStack = True,
                 fitOnStep = True):

        self.trendFinder = TrendFinder()

        #self.levelFeatsScaler = MinMaxlevelFeatsScaler(feature_range=(-1, 1))
        self.levelFeatsScaler = StandardScaler()
        self.closerSpecFeatsScaler = StandardScaler()
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
                                  self.nLevels * (self.nLevels - 1)) + 1 + self.nLevels, )
        else:
            self.featureShape = ( self.nFeatRows, 2*(3 * self.nLevels + 1 + self.nLevels +
                                  self.nLevels * (self.nLevels - 1)) + 1 + self.nLevels )

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

        levelFeatList = []
        closerSpecFeats = []
        trainDfLen = df[currentDate:].shape[0]
        for i in tqdm(range(trainDfLen), desc="Fitting scalers"):
            currentDate = dfDates[currentDateInd]
            feats = self.getOneRowFeat(currentDate, df, None, scaleFeats=False)
            clSpInd = len(feats) - 1 - self.nLevels
            levelFeatList.append( feats[:clSpInd] )
            closerSpecFeats.append( feats[clSpInd:] )
            currentDateInd += 1

        self.levelFeatsScaler.fit( levelFeatList )
        self.closerSpecFeatsScaler.fit( closerSpecFeats )

        return self
    #propose that historyData dataframe has datetime index
    def getFeatByDatetime(self, datetimeStr, historyData, dealOpenPrice):

        obs = None
        if self.nFeatRows > 1:
            obs = self.getManyRowsFeat(datetimeStr, historyData, dealOpenPrice)
        else:
            obs = self.getOneRowFeat(datetimeStr, historyData, dealOpenPrice)

        return obs

    def getOneRowFeat(self, datetimeStr, historyData, dealOpenPrice, scaleFeats=True):

        obsList = historyData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
        #################################
        #scale open and close to (0, 1)
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
        startTime = datetime.now()
        levelFeats = self.trendFinder.getLevelFeatures(obsList, self.nPoints, self.nLevels)

        lastObs = obsList.values[-1]
        lastObs = pd.DataFrame(data=[lastObs], columns=obsList.columns)

        lastAvgPrice = np.average(lastObs[["open", "close"]].values)
        levelFeats = levelFeats + [lastAvgPrice]

        levelPriceDeltas = []
        for i in range(self.nLevels):
            levelPriceDeltas.append( levelFeats[i] - lastAvgPrice )
        levelFeats = levelFeats + levelPriceDeltas

        crossLevelDistances = []
        for i in range(self.nLevels):
            for j in range(self.nLevels):
                if i != j:
                    cld = np.sqrt(np.square(levelFeats[i] - levelFeats[j]))
                    crossLevelDistances.append( cld )
        levelFeats = levelFeats + crossLevelDistances

        ###############################
        volumeFeats = self.getOneRowVolumeFeat(datetimeStr, historyData, scaleFeats=True)
        levelFeats = levelFeats + volumeFeats
        ###############################

        ######################
        #closer key part

        if dealOpenPrice is None:
            minLevel = min(levelFeats[:self.nLevels])
            maxLevel = max(levelFeats[:self.nLevels])
            dealOpenPrice = random.uniform( 0.8 * minLevel, 1.2 * maxLevel)

        closerSpecFeats = []
        lastOpenDelta = lastAvgPrice - dealOpenPrice
        closerSpecFeats.append( lastOpenDelta )

        closerLevelDeltas = []
        for i in range(self.nLevels):
            closerLevelDeltas.append(levelFeats[i] - dealOpenPrice)
        closerSpecFeats = closerSpecFeats + closerLevelDeltas
        #####################

        if scaleFeats:
            levelFeats = np.array(levelFeats).reshape((1, -1))
            levelFeats = self.levelFeatsScaler.transform(levelFeats)
            levelFeats = levelFeats.reshape((-1,))

            closerSpecFeats = np.array(closerSpecFeats).reshape((1, -1))
            closerSpecFeats = self.closerSpecFeatsScaler.transform(closerSpecFeats)
            closerSpecFeats = closerSpecFeats.reshape((-1,))

        obs = np.array( list(levelFeats) + list(closerSpecFeats) )
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


    def getManyRowsFeat(self, datetimeStr, historyData, dealOpenPrice ):

        def processBatch(obsBatch, dealOpenPrice):
            processedObs = []
            for ob in obsBatch:
                lastDt = ob.index.values[-1]
                tmp = self.getOneRowFeat(lastDt, ob, dealOpenPrice, scaleFeats=True)
                processedObs.append(tmp)
            return processedObs

        lastData = historyData.loc[:str(datetimeStr)].tail(self.nPoints + self.nFeatRows - 1)
        lastData.reset_index(inplace=True)

        obsLists = []
        for i in range(self.nFeatRows):
            obsList = lastData[i:i+self.nPoints]
            obsLists.append(obsList)
        for i in range(len(obsLists)):
            obsLists[i].set_index("datetime", drop=True, inplace=True)

        n_jobs = 4
        batchSize = self.nFeatRows // n_jobs
        batches = []
        for i in range(batchSize - 1):
            batches.append( obsLists[i * batchSize : (i+1) * batchSize] )
        batches.append(obsLists[(batchSize-1) * batchSize: ])

        dealPrices = [ dealOpenPrice for i in range(n_jobs)]
        obsLists = Parallel(n_jobs=n_jobs)(delayed(processBatch)(batch, price) for batch, price in zip(batches, dealPrices))
        obs = np.vstack(obsLists)



        return obs

    def getMinDate(self, historyData):

        df = historyData.copy()
        iter = df.iterrows()
        obs = next(iter)
        for i in range(self.nPoints + self.nFeatRows - 1):
            obs = next(iter)

        return obs[0]