from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
import matplotlib.pyplot as plt
import joblib
import numpy as np
from scipy.stats import spearmanr, boxcox
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import *

nExperiments = 3
timeframe = "H1"
symbolList = [ "AUDUSD_i"]
priceFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]
spreadDict = { "EURUSD_i": 18, "AUDUSD_i": 18, "GBPUSD_i": 18, "USDCHF_i": 18, "USDCAD_i": 18 }
spreadCoefDict = { "EURUSD_i": 0.00001,"AUDUSD_i": 0.00001, "GBPUSD_i": 0.00001, "USDCHF_i": 0.00001, "USDCAD_i": 0.00001  }
#terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()
################################################

def getCumulativeReward(dealsStatistics):
    sumRew = 0
    cumulativeReward = []
    for i in range(len(dealsStatistics)):
            sumRew += dealsStatistics[i]
            cumulativeReward.append(sumRew)
    return cumulativeReward

def convertToLinearSet(dealsStatistics):
    dealsStatistics = np.reshape(dealsStatistics, (-1,))
    X = np.array([x for x in range(len(dealsStatistics))])
    X = np.reshape(X, (-1, 1))
    linReg = HuberRegressor().fit(X, dealsStatistics)
    tmp = linReg.predict(X)
    #plt.plot(X, tmp)
    #plt.plot(X, dealsStatistics)
    #plt.show()
    return tmp

def loadDealsStatistics(symbol, nExperiment):
    testDealsStatistics = None
    backDealsStatistics = None
    trainDealsStatistics = None
    with open("./" + "testDealsStatistics_{}_{}.pkl".format(symbol, nExperiment), mode="rb") as dealsFile:
        testDealsStatistics = joblib.load(dealsFile)
    with open("./" + "backDealsStatistics_{}_{}.pkl".format(symbol, nExperiment), mode="rb") as dealsFile:
        backDealsStatistics = joblib.load(dealsFile)
    with open("./" + "trainDealsStatistics_{}_{}.pkl".format(symbol, nExperiment), mode="rb") as dealsFile:
        trainDealsStatistics = joblib.load(dealsFile)
    dealsStatistics = []
    dealsStatistics.append(testDealsStatistics)
    dealsStatistics.append(trainDealsStatistics)
    dealsStatistics.append(backDealsStatistics)
    return dealsStatistics

corrCoefPacks = []
for symbol in symbolList:
    for nExperiment in range(nExperiments):
        dealsStatistics = loadDealsStatistics(symbol, nExperiment)
        cumRewList = []
        for seriesPack in dealsStatistics:
            cumRewSeriesPack = []

            #check length
            nullLen = False
            for dealsPack in seriesPack:
                packLen = len(dealsPack)
                if packLen == 0:
                    nullLen = True
                    break
            if nullLen:
                continue

            for dealsPack in seriesPack:
                localCumRew = getCumulativeReward(dealsPack)
                localCumRew = convertToLinearSet(localCumRew)
                cumRewSeriesPack.append( localCumRew )
                #cumRewSeriesPack.append(dealsPack)
            cumRewList.append(cumRewSeriesPack)
        ttCorrArr = []
        tbCorrArr = []
        btCorrArr = []
        for i in range(len(cumRewList[0])):
            testCumRew = cumRewList[0][i]
            trainCumRew = cumRewList[1][i]
            backCumRew = cumRewList[2][i]
            #testCumRew = MinMaxScaler(feature_range=(1, 100)).fit_transform(np.reshape(testCumRew, newshape=(-1, 1)))
            #trainCumRew = MinMaxScaler(feature_range=(1, 100)).fit_transform(np.reshape(trainCumRew, newshape=(-1, 1)))
            #backCumRew = MinMaxScaler(feature_range=(1, 100)).fit_transform(np.reshape(backCumRew, newshape=(-1, 1)))
            #testCumRew = np.reshape(testCumRew, newshape=(1, -1))[0]
            #trainCumRew = np.reshape(trainCumRew, newshape=(1, -1))[0]
            #backCumRew = np.reshape(backCumRew, newshape=(1, -1))[0]
            #testCumRew = boxcox(testCumRew)[0]
            #trainCumRew = boxcox(trainCumRew)[0]
            #backCumRew = boxcox(backCumRew)[0]

            minLen = min( [len(testCumRew), len(backCumRew)] )
            tmpTrain = np.sort(np.random.choice(trainCumRew, size=minLen))
            tmpTest = np.sort(np.random.choice(testCumRew, size=minLen))
            tmpBack = np.sort(np.random.choice(backCumRew, size=minLen))
            #for j in range(1000):
            #    tmpTrain = np.hstack( [ tmpTrain, np.random.choice(trainCumRew, size=minLen)] )
            #    tmpTest = np.hstack( [ tmpTest, np.random.choice(testCumRew, size=minLen)] )
            #    tmpBack = np.hstack( [ tmpBack, np.random.choice(backCumRew, size=minLen)] )

            """maxLen = max([len(testCumRew), len(backCumRew), len(trainCumRew)])
            tmpTrain = trainCumRew
            #tmpTest = np.random.choice(testCumRew, size=maxLen, replace=True)
            #tmpBack = np.random.choice(backCumRew, size=maxLen, replace=True)
            nRep = len(tmpTrain) // len(testCumRew)
            tailElem = len(trainCumRew) - nRep * len(testCumRew)
            tmpTest = testCumRew
            for j in range(nRep-1):
                tmpTest = np.hstack( [ tmpTest, testCumRew] )
            tmpTest = np.hstack( [ tmpTest, testCumRew[:tailElem]] )

            nRep = len(tmpTrain) // len(backCumRew)
            tailElem = len(trainCumRew) - nRep * len(backCumRew)
            tmpBack = backCumRew
            for j in range(nRep-1):
                tmpBack = np.hstack([tmpBack, backCumRew])
            tmpBack = np.hstack([tmpBack, backCumRew[:tailElem]])"""



            #tmpTrain = np.random.choice(trainCumRew, size=len(testCumRew))
            testTrainCorr = spearmanr(tmpTrain, tmpTest).correlation

            #tmpTrain = np.random.choice(trainCumRew, size=len(backCumRew))
            trainBackCorr = spearmanr(tmpTrain, tmpBack).correlation

            backTestCorr = spearmanr(tmpBack, tmpTest).correlation

            ##############################################################
            """testCumRew = MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.reshape(testCumRew, newshape=(-1, 1)))
            trainCumRew = MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.reshape(trainCumRew, newshape=(-1, 1)))
            backCumRew = MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.reshape(backCumRew, newshape=(-1, 1)))
            testCumRew = np.reshape(testCumRew, newshape=(1, -1))[0]
            trainCumRew = np.reshape(trainCumRew, newshape=(1, -1))[0]
            backCumRew = np.reshape(backCumRew, newshape=(1, -1))[0]"""

            #testTrainCorr = np.std(testCumRew)
            #trainBackCorr = np.std(trainCumRew)
            #backTestCorr = np.std(backCumRew)

            #testTrainCorr = np.mean(testCumRew)
            #trainBackCorr = np.mean(trainCumRew)
            #backTestCorr = np.mean(backCumRew)

            #testTrainCorr = testCumRew[-1]
            #trainBackCorr = trainCumRew[-1]
            #backTestCorr = backCumRew[-1]

            ttCorrArr.append(testTrainCorr)
            tbCorrArr.append(trainBackCorr)
            btCorrArr.append(backTestCorr)
        corrCoefPacks.append([ttCorrArr, tbCorrArr, btCorrArr])

for i in range(len(corrCoefPacks)):
    for j in range( len(corrCoefPacks[i]) ):
        plt.plot( [x for x in range(len(corrCoefPacks[i][j]))], corrCoefPacks[i][j], label="{}_{}".format(i,j) )
plt.legend()
plt.show()