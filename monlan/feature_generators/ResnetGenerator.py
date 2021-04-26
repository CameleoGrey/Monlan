
from sklearn.preprocessing import StandardScaler
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
from monlan.feature_generators.ResnetPredictor import ResnetPredictor

class ResnetGenerator():
    def __init__(self , featureList=[], nPoints = 32, nDiffs=1):

        self.scalerDict = {}
        for feat in featureList:
            self.scalerDict[feat] = StandardScaler()
        self.featureList = featureList
        self.featureListSize = len( featureList )
        self.nPoints = nPoints
        self.nDiffs = nDiffs
        self.iter = iter
        self.featureShape = (self.featureListSize,)

        self.fitMode = False
        self.fitData = None

        self.predictor = ResnetPredictor()
        pass

    def setFitMode(self, fitMode):
        if fitMode == True:
            self.fitMode = True
        else:
            self.fitMode = False
            self.fitData = None
        pass

    def fitPredictor(self, dfList, featureList, batch_size, epochs, lr=0.001, verbose=1):
        trainX, testX, trainY, testY = self.predictor.makeTrainTestDataSets(dfList=dfList, nDiffs=1,
                                                                        featureList=featureList)

        self.predictor.build_model(inputShape=(self.nPoints, self.featureListSize, 1),
                                   outputShape=(self.featureListSize,), lr=lr)
        self.predictor.fit(trainX, trainY, validation_data=(testX, testY),
                           batch_size=batch_size, epochs=epochs, verbose=verbose)
        pass

    def prepareGenerator(self, df):
        dfCopy = df.copy()

        preprocDf = self.predictor.preprocDf(dfCopy, self.nDiffs, self.featureList, returnScalers=False)

        self.fitData = self.buildFitModeData(preprocDf)
        for feat in self.featureList:
            featVals = self.fitData[feat].values
            featVals = np.reshape(featVals, (-1, 1))
            self.scalerDict[feat].fit( featVals )
            featVals = self.scalerDict[feat].transform(featVals)
            featVals = np.reshape( featVals, (-1,) )
            self.fitData[feat] = featVals
        self.fitData.set_index("datetime", drop=True, inplace=True)

        pass

    def buildFitModeData(self, preprocDf):

        preprocDf = preprocDf.copy()
        for colName in preprocDf.columns:
            if colName not in self.featureList and colName not in ["datetime"]:
                del preprocDf[colName]

        x = []
        dfVals = preprocDf[self.featureList]
        dfVals = dfVals.values
        #dfVals = dfVals[:, 1:] #do not touch "datetime" column
        for i in range(len(dfVals) - self.nPoints):
            x.append(dfVals[i:i + self.nPoints])
        x = np.expand_dims(x, axis=-1)

        predicts = self.predictor.predict(x)
        preprocDf = preprocDf.tail( len(preprocDf) - self.nPoints )
        colNames = list(preprocDf.columns)
        for i in range(len(self.featureList)):
            preprocDf[self.featureList[i]] = predicts[:, i]
        fitModeData = preprocDf

        #acceleration
        """for i in range(1):
            for feat in self.featureList:
                notShifted = fitModeData[feat]
                shiftedData = fitModeData[feat].shift(periods=1)
                fitModeData[feat] = notShifted - shiftedData
            iter = next(fitModeData.iterrows())
            fitModeData = fitModeData.drop(iter[0])"""

        return fitModeData

    #propose that historyData dataframe has datetime index
    def getFeatByDatetime(self, datetimeStr, historyData, expandDims=True):

        obs = None
        if self.fitMode:
            #obs = self.fitData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
            #obs = np.array(obs)
            #obs = np.reshape(obs, (obs.shape[0], obs.shape[1], 1))

            obs = self.fitData.loc[str(datetimeStr)].copy()
            obs = np.array(obs)
            #obs = np.reshape(obs, (obs.shape[0], 1))

        if expandDims:
            obs = np.expand_dims(obs, axis=0)

        return obs

    def getManyPointsFeat(self, datetimeStr, historyData):

        obsList = None
        if self.fitMode:
            obsList = self.fitData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
        else:
            df = historyData.copy()

            ########
            df = df[:str(datetimeStr)].tail(500)
            from monlan.mods.VSASpread import VSASpread
            from monlan.mods.HeikenAshiMod import HeikenAshiMod
            from monlan.mods.EnergyMod import EnergyMod
            historyMod = VSASpread()
            df = historyMod.modHistory(df)
            historyMod = HeikenAshiMod()
            df = historyMod.modHistory(df)
            historyMod = EnergyMod()
            df = historyMod.modHistory(df, featList=["open", "close", "low", "high"])
            ########

            for i in range(self.nDiffs):
                for feat in self.featureList:
                    notShifted = df[feat]
                    shiftedData = df[feat].shift(periods=1)
                    df[feat] = notShifted - shiftedData
                iter = next(df.iterrows())
                df = df.drop(iter[0])

            obsList = df.loc[:str(datetimeStr)].tail(self.nPoints).copy()

            obsListTransformed = {}
            for obs in obsList.iterrows():
                for feat in self.featureList:
                    data = obs[1][feat]
                    data = data.reshape(-1, 1)
                    data = self.scalerDict[feat].transform(data)
                    obs[1][feat] = data
                obsListTransformed[obs[0]] = obs[1]
            obsList = pd.DataFrame(obsListTransformed.values(), index=list(obsListTransformed.keys()))

        obs = []
        if self.flatStack:
            localObs = np.zeros((self.nPoints,))
            # for feat in self.featureList:
            #    localObs = np.vstack([localObs, vectorPrices[feat]])
            # obs = np.array(localObs[1:])
        else:
            localObs = np.zeros((self.nPoints,))
            for feat in self.featureList:
                localObs = np.vstack([localObs, obsList[feat].values])
            obs.append(localObs[1:])
        obs = np.array(obs)

        obs = np.reshape(obs, (obs.shape[1], obs.shape[2], 1))

        return obs

    def getMinDate(self, historyData):
        df = historyData.copy()

        ########
        from monlan.mods.VSASpread import VSASpread
        from monlan.mods.HeikenAshiMod import HeikenAshiMod
        from monlan.mods.EnergyMod import EnergyMod
        historyMod = VSASpread()
        df = historyMod.modHistory(df)
        historyMod = HeikenAshiMod()
        df = historyMod.modHistory(df)
        historyMod = EnergyMod()
        df = historyMod.modHistory(df, featList=["open", "close", "low", "high"])
        ########

        for i in range(self.nDiffs):
            for feat in self.featureList:
                notShifted = df[feat]
                shiftedData = df[feat].shift(periods=1)
                df[feat] = notShifted - shiftedData
            iter = next(df.iterrows())
            df = df.drop(iter[0])
        iter = df.iterrows()
        obs = next(iter)
        #acceleration
        #for i in range(self.nPoints + self.nPoints):
        #
        for i in range(self.nPoints - 1):
            obs = next(iter)

        return obs[0]

    def saveGenerator(self, dir, name):
        self.predictor.save(dir, name + "_predictor")
        tmp = self.predictor
        self.predictor = None
        with open(dir + name + "_object.pkl", 'wb') as genFile:
            joblib.dump(self, genFile)
        self.predictor = tmp
        print("Generator saved: {}".format(dir + name))
        pass

    def loadGenerator(self, dir, name):
        generator = None
        with open(dir + name + "_object.pkl", "rb") as genFile:
            generator = joblib.load(genFile)
        predictor = ResnetPredictor().load(dir, name + "_predictor")
        generator.predictor = predictor
        print("Generator loaded: {}".format(dir + name))
        return generator