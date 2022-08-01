"""
ResNet + MultiScaler + extended rows
"""
from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
from avera.envs.RealCompositeEnv import RealCompositeEnv
from avera.feature_generators.CompositeGenerator import CompositeGenerator
from avera.feature_generators.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.terminal.MT5Terminal import MT5Terminal
from avera.mods.HeikenAshiMod import HeikenAshiMod
from avera.mods.EnergyMod import EnergyMod
from avera.mods.VSASpread import VSASpread
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from avera.utils.keras_utils import useCpu, reset_keras
import numpy as np

def preprocData(df):
    historyMod = VSASpread()
    modDf = historyMod.modHistory(df)
    historyMod = HeikenAshiMod()
    modDf = historyMod.modHistory(modDf)
    historyMod = EnergyMod()
    modDf = historyMod.modHistory(modDf, featList=["open", "close", "low", "high"])
    return modDf

def createGenerator( df, featureList, saveDir, saveName ):
    priceDiffGenerator = MultiScalerDiffGenerator(featureList=featureList, nDiffs=1, nPoints=32, flatStack=False,
                                                  fitOnStep=False)
    priceDiffGenerator.setFitMode(True)
    priceDiffGenerator = priceDiffGenerator.globalFit(df)
    priceDiffGenerator.saveGenerator("{}{}.pkl".format(saveDir, saveName))

def trainAgent(trainDf, backTestDf, genPath, saveAgentDir, saveAgentName):
    startTime = datetime.now()
    print("Start train time: {}".format(startTime))

    openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("{}".format(genPath))
    buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("{}".format(genPath))
    sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("{}".format(genPath))

    trainEnv = CompositeEnv(trainDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                            startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                            stoplossPuncts=100, takeprofitPuncts=300,
                            renderFlag=True, renderDir=saveDir, renderName="trainDealsPlot")
    backTestEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                               startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                               stoplossPuncts=100, takeprofitPuncts=300,
                               renderFlag=True, renderDir=saveDir, renderName="backTestDealsPlot")
    # get size of state and action from environment
    openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=2500, batch_size=100, train_start=200, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=0.9997, learning_rate=0.0002)
    buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                          epsilon_decay=1.0, learning_rate=0.0002)
    sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=1.0, learning_rate=0.0002)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=4, nWarmUp=0,
                                 uniformEps=False, synEps=True, plotScores=False, saveBest=True, saveFreq=1,
                                 saveDir = saveAgentDir, saveName = saveAgentName)

    endTime = datetime.now()
    print("Training finished. Total time: {}".format(endTime - startTime))
    return lastSaveEp

def evaluateAgent(backTestDf, genPath, agentDir, agentName):

    openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("{}".format(genPath))
    buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("{}".format(genPath))
    sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("{}".format(genPath))

    testEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                           startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                           stoplossPuncts=100, takeprofitPuncts=300,
                           renderFlag=True, renderDir=saveDir, renderName="testDealsPlot")

    # get size of state and action from environment
    openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
    buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
    sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    agent = agent.load_agent(agentDir, agentName)
    # agent  = agent.load_agent("./", "checkpoint_composite")
    print("start using agent")
    dealsStatistics = agent.use_agent(testEnv)
    sumRew = 0
    for i in range(len(dealsStatistics)):
        sumRew += dealsStatistics[i]
    return sumRew

def cycleTrain(symbol, timeframe, dataUpdater, saveDir, genPath, saveAgentName):
    while True:
        dataUpdater.partialUpdate(terminal, symbol, timeframe)
        df = SymbolDataManager().getData(symbol, timeframe)
        preprocDf = preprocData(df)
        createGenerator(preprocDf, featureList=hkFeatList, saveDir=saveDir, saveName=genName)

        trainDf = preprocDf.tail(2032).tail(1032)
        backTestDf = preprocDf.tail(2032).head(1032)

        lastSaveEp = trainAgent(trainDf=trainDf, backTestDf=backTestDf,
                                genPath=genPath, saveAgentDir=saveDir, saveAgentName=saveAgentName)
        ##############
        reset_keras()
        ##############

        #####
        # break
        #####

        sumReward = evaluateAgent(backTestDf=backTestDf, genPath=genPath, agentDir=saveDir, agentName=saveAgentName)
        if lastSaveEp > 0 and sumReward > -110:
            print("Backtest's succesfull. Stop Training.")
            break
        else:
            print("Backtest failed. Retrain.")

        #############
        #weighted train/back reward
        """backReward = evaluateAgent(backTestDf=backTestDf, genPath=genPath, agentDir=saveDir, agentName=saveAgentName)
        trainReward = evaluateAgent(backTestDf=trainDf, genPath=genPath, agentDir=saveDir, agentName=saveAgentName)
        backW = 0.8 #1.0 - (len(backTestDf) / (len(backTestDf) + len(trainDf)))
        trainW = 0.2 #1.0 - (len(trainDf) / (len(backTestDf) + len(trainDf)))
        weightedReward = (backW * backReward + trainW * trainReward) / 2
        if lastSaveEp > 0 and weightedReward > 0:
            print("Backtest's succesfull. Stop Training.")
            break
        else:
            print("Backtest failed. Retrain.")"""
        #############

        reset_keras()

def useAgent(symbol, timeframe, terminal, dataUpdater, dataManager,
             hkFeatList, saveDir, genPath, agentName, timeConstraint):
    ###############################
    # use agent on test
    ###############################

    useCpu(nThreads=8, nCores=8)

    openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator(genPath)
    buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator(genPath)
    sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator(genPath)

    openerPriceDiffGenerator.setFitMode(False)
    buyerPriceDiffGenerator.setFitMode(False)
    sellerPriceDiffGenerator.setFitMode(False)

    testEnv = RealCompositeEnv(symbol, timeframe, terminal, dataUpdater, dataManager,
                               openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                               startDeposit=300, lotSize=0.1, lotCoef=100000,
                               stoplossPuncts=60, takeprofitPuncts=120, renderFlag=True)

    # get size of state and action from environment
    openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
    buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
    sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    agent = agent.load_agent(saveDir, agentName, dropSupportModel=True)
    # agent  = agent.load_agent("./", "checkpoint_composite")
    print("start using agent")

    startTime = datetime.now()
    agent.use_agent(testEnv, timeConstraint=timeConstraint)
    endTime = datetime.now()
    print("Use time: {}".format(endTime - startTime))
    reset_keras()

def selectSymbol(symbolList, dataManager, period):
    featVals = []
    for sym in symbolList:
        df = dataManager.getData(sym, timeframe)
        df = df.tail(period)
        #feat = np.abs( np.abs(df["close"].values) - np.abs(df["open"].values) )
        #feat = abs(np.mean(feat))

        feat = 100 * (np.max(df["close"].values) / np.min(df["open"].values) - 1)
        feat = round(feat, 3)

        #feat = df["close"].values
        #feat = np.diff(feat)
        #feat = int( np.sum(np.abs(feat)) * 100000 )

        featVals.append(feat)
    featDict = {}
    for i in range(len(symbolList)):
        featDict[symbolList[i]] = featVals[i]
    print(featDict)

    selectedSymbol = symbolList[ np.argmax(featVals) ]
    return selectedSymbol

"""def selectSymbol(symbolList, dataManager, period):
    def getCurveLen(y):
        from scipy.integrate import simps
        import matplotlib.pyplot as plt
        #y = y / np.sqrt( np.sum(np.square(y)) )
        y = np.diff(y)
        tmp = []
        for i in range(y.shape[0]):
            tmp.append( np.sqrt(1 + y[i]**2) )
        y = np.asarray(tmp)
        x = [x for x in range(y.shape[0])]
        curveLen = simps(y, x)
        return curveLen
    featVals = []
    for sym in symbolList:
        df = dataManager.getData(sym, timeframe)
        df = df.tail(period + 1)
        feat = (df["open"].values + df["close"].values + df["low"].values + df["high"].values) / 4
        feat = getCurveLen(feat)
        #feat = (feat - period + 1) * 100000
        featVals.append(feat)
    featDict = {}
    for i in range(len(symbolList)):
        featDict[symbolList[i]] = featVals[i]
    print(featDict)

    selectedSymbol = symbolList[ np.argmax(featVals) ]
    return selectedSymbol"""

####################################################################################

symbol = "EURUSD_i"
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]
saveDir = "../models/"
genName = "MSDiffGen"
genPath = saveDir + genName + ".pkl"
agentName = "best_composite"
timeConstraint = timedelta(days=5, hours=12, minutes=0)

terminal = MT5Terminal(login=123456, server="broker-server", password="password")
#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()

dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
while True:
    #################
    """symbolList = ["EURUSD_i", "USDCHF_i", "USDCAD_i", "AUDUSD_i"]
    for sym in symbolList:
        print("Partial update: {}".format(sym))
        #dataUpdater.fullUpdate(terminal, sym, timeframe, startDate="2015-01-01 00:00:00")
        dataUpdater.partialUpdate(terminal, sym, timeframe)
    symbol = selectSymbol(symbolList, dataManager, period=18)
    print("Selected symbol: {}".format(symbol))"""
    #################
    #cycleTrain(symbol, timeframe, dataUpdater, saveDir, genPath, agentName)
    useAgent(symbol, timeframe, terminal, dataUpdater, dataManager,
             hkFeatList, saveDir, genPath, agentName, timeConstraint)
    print("Use period finished. Going retrain.")