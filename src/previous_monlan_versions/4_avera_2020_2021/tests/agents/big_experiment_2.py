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

def getTrainTestSets(df, nExperiments, mainPackSize=10000):
    trainTestSets = []
    dataStep = df.shape[0] // nExperiments
    for i in range(nExperiments):
        dataPart = df[i*dataStep : (i+1)*dataStep]
        dataPart = dataPart.tail(mainPackSize)
        #trainSet = dataPart[ : int(trainSize * dataPart.shape[0]) ]
        trainSet = dataPart.tail(6200).head(5200)
        #testSet = dataPart[ int(trainSize * dataPart.shape[0]) : ]
        testSet = dataPart.tail(6200).tail(1200)
        backTestSet = dataPart.tail(7200).head(1200)
        trainTestSets.append( [trainSet, testSet, backTestSet] )
    return trainTestSets

def createGenerators(df, priceFeatList):
    priceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList, nDiffs=1, nPoints=200, flatStack=False,
                                                  fitOnStep=False)
    priceDiffGenerator.setFitMode(True)
    priceDiffGenerator = priceDiffGenerator.globalFit(df)
    priceDiffGenerator.saveGenerator("./MSDiffGen.pkl")

    pass

def trainAgent(trainDf, backTestDf, startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001 ):
    openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./MSDiffGen.pkl")
    buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./MSDiffGen.pkl")
    sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./MSDiffGen.pkl")

    trainEnv = CompositeEnv(trainDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                            startDeposit=startDeposit, lotSize=lotSize, lotCoef=lotCoef, spread=spread, spreadCoef=spreadCoef, renderFlag=True)
    backTestEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                               startDeposit=startDeposit, lotSize=lotSize, lotCoef=lotCoef, spread=spread, spreadCoef=spreadCoef, renderFlag=True)
    # get size of state and action from environment
    openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=2500, batch_size=100, train_start=200, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=0.9999, learning_rate=0.0002)
    buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                          epsilon_decay=1.0, learning_rate=0.0002)
    sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=1.0, learning_rate=0.0002)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=3, nWarmUp=0,
                    uniformEps=False, synEps=True, plotScores=True, saveBest=False, saveFreq=1)
    return lastSaveEp

def collectStatistics(testDf, startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001 ):
    openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./MSDiffGen.pkl")
    buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./MSDiffGen.pkl")
    sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./MSDiffGen.pkl")

    #openerPriceDiffGenerator.setFitMode(False)
    #buyerPriceDiffGenerator.setFitMode(False)
    #sellerPriceDiffGenerator.setFitMode(False)

    testEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                            startDeposit=startDeposit, lotSize=lotSize, lotCoef=lotCoef, spread=spread, spreadCoef=spreadCoef, renderFlag=True)
    openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
    buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
    sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    #agent  = agent.load_agent("./", "best_composite")
    agent = agent.load_agent("./", "checkpoint_composite")
    print("start using agent")
    dealsStatistics = agent.use_agent(testEnv)
    return dealsStatistics
#########################################################################################################
timeframe = "H1"
symbolList = [ "USDCAD_i"]
priceFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]
spreadDict = { "EURUSD_i": 18, "AUDUSD_i": 18, "GBPUSD_i": 18, "USDCHF_i": 18, "USDCAD_i": 18 }
spreadCoefDict = { "EURUSD_i": 0.00001,"AUDUSD_i": 0.00001, "GBPUSD_i": 0.00001, "USDCHF_i": 0.00001, "USDCAD_i": 0.00001  }
terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()
###########################################################################################################

for symbol in symbolList:
    dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")

saveEpInfo = []
for symbol in symbolList:
    df = SymbolDataManager().getData(symbol, timeframe)
    ########
    historyMod = VSASpread()
    df = historyMod.modHistory(df)
    historyMod = HeikenAshiMod()
    df = historyMod.modHistory(df)
    historyMod = EnergyMod()
    df = historyMod.modHistory(df, featList=["open", "close", "low", "high"])
    ########
    createGenerators(df, priceFeatList)

    trainTestSets = getTrainTestSets(df, nExperiments=3, mainPackSize=10400)
    trainDealsStatistics = []
    testDealsStatistics = []
    backDealsStatistics = []
    nStep = 0
    for trainX, testX, backTestX in trainTestSets:
        nStep += 1
        print("Experiment №{}".format(nStep))
        startTime = datetime.now()
        print("start time: {}".format(startTime))
        lastSaveEp = trainAgent(trainX.copy(), backTestX.copy(), spread=spreadDict[symbol], spreadCoef=spreadCoefDict[symbol])
        endTime = datetime.now()
        print("Training finished. Total time: {}".format(endTime - startTime))
        saveEpInfo.append([symbol, nStep, lastSaveEp])
        testDealsStatistics.append( collectStatistics(testX, spread=spreadDict[symbol], spreadCoef=spreadCoefDict[symbol]) )
        backDealsStatistics.append( collectStatistics(backTestX, spread=spreadDict[symbol], spreadCoef=spreadCoefDict[symbol]) )
        with open("./" + "saveEpInfo.pkl", mode="wb") as saveEpFile:
            joblib.dump(saveEpInfo, saveEpFile)
        with open("./" + "testDealsStatistics_{}_{}.pkl".format(symbol, timeframe), mode="wb") as dealsFile:
            joblib.dump(testDealsStatistics, dealsFile)
        with open("./" + "backDealsStatistics_{}_{}.pkl".format(symbol, timeframe), mode="wb") as dealsFile:
            joblib.dump(backDealsStatistics, dealsFile)
print("statistics collected")

with open("./" + "saveEpInfo.pkl", mode="rb") as saveEpFile:
    saveEpInfo = joblib.load(saveEpFile)
    print(saveEpInfo)

for symbol in symbolList:
    testDealsStatistics = None
    with open("./" + "testDealsStatistics_{}_{}.pkl".format(symbol, timeframe), mode="rb") as dealsFile:
        testDealsStatistics = joblib.load(dealsFile)
    for j in range(len(testDealsStatistics)):
        sumRew = 0
        cumulativeReward = []
        for i in range(len(testDealsStatistics[j])):
            sumRew += testDealsStatistics[j][i]
            cumulativeReward.append(sumRew)
        plt.plot( [x for x in range(len(cumulativeReward))], cumulativeReward, label=symbol + "_" + str(j) )
        plt.legend()
plt.show()

for symbol in symbolList:
    backDealsStatistics = None
    with open("./" + "backDealsStatistics_{}_{}.pkl".format(symbol, timeframe), mode="rb") as dealsFile:
        backDealsStatistics = joblib.load(dealsFile)
    for j in range(len(backDealsStatistics)):
        sumRew = 0
        cumulativeReward = []
        for i in range(len(backDealsStatistics[j])):
            sumRew += backDealsStatistics[j][i]
            cumulativeReward.append(sumRew)
        plt.plot( [x for x in range(len(cumulativeReward))], cumulativeReward, label= symbol + "_" + str(j) )
        plt.legend()
plt.show()
