"""
collect statistics to train model selector
"""

from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
from avera.envs.RealCompositeEnv import RealCompositeEnv
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from avera.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from avera.feature_generators.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.mods.HeikenAshiMod import HeikenAshiMod
from avera.mods.EnergyMod import EnergyMod
from avera.mods.VSASpread import VSASpread
from avera.terminal.MT5Terminal import MT5Terminal
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
from scipy.stats import spearmanr
from avera.utils.keras_utils import reset_keras


def getTrainTestSets(df, nExperiments):
    trainTestSets = []
    dataStep = df.shape[0] // nExperiments
    for i in range(nExperiments):
        dataPart = df[i*dataStep : (i+1)*dataStep]
        trainSet = dataPart.tail(1132).head(1032)
        testSet = dataPart.tail(1132).tail(132)
        backTestSet = dataPart.tail(1232).head(132)
        trainTestSets.append( [trainSet, testSet, backTestSet] )
    return trainTestSets

def createGenerators(df, priceFeatList):
    ########
    historyMod = VSASpread()
    df = historyMod.modHistory(df)
    historyMod = HeikenAshiMod()
    df = historyMod.modHistory(df)
    historyMod = EnergyMod()
    df = historyMod.modHistory(df, featList=["open", "close", "low", "high"])
    ########

    priceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList, nDiffs=1, nPoints=32, flatStack=False,fitOnStep=False)
    priceDiffGenerator.setFitMode(True)
    priceDiffGenerator = priceDiffGenerator.globalFit(df)
    priceDiffGenerator.saveGenerator("./MSDiffGen.pkl")

    pass

def collectStatistics(symbol, nExperiment, trainDf, backTestDf, testDf, startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001):
    reset_keras()
    openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./MSDiffGen.pkl")
    buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./MSDiffGen.pkl")
    sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./MSDiffGen.pkl")

    trainEnv = CompositeEnv(trainDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                            startDeposit=startDeposit, lotSize=lotSize, lotCoef=lotCoef, spread=spread, spreadCoef=spreadCoef, renderFlag=True)
    backTestEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                               startDeposit=startDeposit, lotSize=lotSize, lotCoef=lotCoef, spread=spread, spreadCoef=spreadCoef, renderFlag=True)
    forwardTestEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                               startDeposit=startDeposit, lotSize=lotSize, lotCoef=lotCoef, spread=spread, spreadCoef=spreadCoef, renderFlag=True)
    # get size of state and action from environment
    openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=2500, batch_size=100, train_start=200, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=0.9995, learning_rate=0.0005)
    buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                          epsilon_decay=1.0, learning_rate=0.0005)
    sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=1.0, learning_rate=0.0005)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)

    continueFit = False
    trainDealsStatistics = []
    testDealsStatistics = []
    backDealsStatistics = []
    for i in range(20):
        agent.fit_agent(env=trainEnv, backTestEnv=None, nEpisodes=1, nWarmUp=0,
                        uniformEps=False, synEps=True, plotScores=True, saveBest=False, saveFreq=111, continueFit=continueFit)
        continueFit = True

        trainDealsStatistics.append( agent.use_agent(trainEnv) )
        testDealsStatistics.append( agent.use_agent(forwardTestEnv) )
        backDealsStatistics.append( agent.use_agent(backTestEnv) )

        with open("./" + "trainDealsStatistics_{}_{}.pkl".format(symbol, nExperiment), mode="wb") as dealsFile:
            joblib.dump(trainDealsStatistics, dealsFile)
        with open("./" + "testDealsStatistics_{}_{}.pkl".format(symbol, nExperiment), mode="wb") as dealsFile:
            joblib.dump(testDealsStatistics, dealsFile)
        with open("./" + "backDealsStatistics_{}_{}.pkl".format(symbol, nExperiment), mode="wb") as dealsFile:
            joblib.dump(backDealsStatistics, dealsFile)
    pass

#########################################################################################################
nExperiments = 20
timeframe = "H1"
symbolList = [ "USDCAD_i"]
priceFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]
spreadDict = { "EURUSD_i": 18, "AUDUSD_i": 18, "GBPUSD_i": 18, "USDCHF_i": 18, "USDCAD_i": 18 }
spreadCoefDict = { "EURUSD_i": 0.00001,"AUDUSD_i": 0.00001, "GBPUSD_i": 0.00001, "USDCHF_i": 0.00001, "USDCAD_i": 0.00001  }
terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")
###########################################################################################################
for symbol in symbolList:
    dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")

saveEpInfo = []
for symbol in symbolList:
    df = SymbolDataManager().getData(symbol, timeframe)
    createGenerators(df, priceFeatList)

    nStep = 0
    trainTestSets = getTrainTestSets(df, nExperiments=nExperiments)
    for trainX, testX, backTestX in trainTestSets:
        print("Experiment №{}".format(nStep))
        startTime = datetime.now()
        print("start time: {}".format(startTime))
        collectStatistics( symbol=symbol, nExperiment=nStep, startDeposit=300, lotSize=0.1,
                            trainDf=trainX.copy(), backTestDf=backTestX.copy(), testDf=testX.copy(),
                            spread=spreadDict[symbol], spreadCoef=spreadCoefDict[symbol])
        endTime = datetime.now()
        nStep += 1
        print("Training finished. Total time: {}".format(endTime - startTime))
print("statistics collected")