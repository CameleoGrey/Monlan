"""
ResNet + MultiScaler + extended rows
"""
from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
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
                            renderFlag=True, renderDir=saveDir, renderName="trainDealsPlot")
    backTestEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                               startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                               renderFlag=True, renderDir=saveDir, renderName="backTestDealsPlot")
    # get size of state and action from environment
    openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=2500, batch_size=100, train_start=200, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=0.9994, learning_rate=0.001)
    buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                          epsilon_decay=0.9999, learning_rate=0.001)
    sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=0.9999, learning_rate=0.001)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=5, nWarmUp=0,
                                 uniformEps=False, synEps=True, plotScores=True, saveBest=True, saveFreq=1,
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
                           renderFlag=True)

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

def cycleTrain(saveDir, genPath, saveAgentName):
    while True:

        preprocDf = preprocData(df)
        createGenerator(preprocDf, featureList=hkFeatList, saveDir=saveDir, saveName=genName)

        trainDf = preprocDf.tail(1232).head(1032)
        backTestDf = preprocDf.tail(1432).head(232)

        lastSaveEp = trainAgent(trainDf=trainDf, backTestDf=backTestDf,
                                genPath=genPath, saveAgentDir=saveDir, saveAgentName=saveAgentName)

        #####
        # break
        #####

        sumReward = evaluateAgent(backTestDf=backTestDf, genPath=genPath, agentDir=saveDir, agentName=saveAgentName)
        if lastSaveEp > 0 and sumReward > 0:
            break
        else:
            print("Backtest failed. Retrain.")

        reset_keras()

def useAgent(preprocDf, hkFeatList, saveDir, genPath, agentName, timeConstraint):
    ###############################
    # use agent on test
    ###############################

    useCpu(nThreads=8, nCores=8)

    testDf = preprocDf.tail(232)

    openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator(genPath)
    buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator(genPath)
    sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator(genPath)

    #openerPriceDiffGenerator.setFitMode(False)
    #buyerPriceDiffGenerator.setFitMode(False)
    #sellerPriceDiffGenerator.setFitMode(False)

    testEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                           startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                           renderFlag=True,
                           renderDir=saveDir, renderName="testDealsPlot")

    # get size of state and action from environment
    openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
    buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
    sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    agent = agent.load_agent(saveDir, agentName, dropSupportModel=True)
    # agent  = agent.load_agent("./", "checkpoint_composite")
    print("start using agent")

    startTime = datetime.now()
    dealsStatistics = agent.use_agent(testEnv, timeConstraint=timeConstraint)
    endTime = datetime.now()
    print("Use time: {}".format(endTime - startTime))
    reset_keras()

    sumRew = 0
    cumulativeReward = []
    for i in range(len(dealsStatistics)):
        sumRew += dealsStatistics[i]
        cumulativeReward.append(sumRew)
    plt.plot([x for x in range(len(cumulativeReward))], cumulativeReward)
    plt.show()

symbol = "EURUSD_i"
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]
saveDir = "../models/"
genName = "saved_gen"
genPath = saveDir + genName + ".pkl"
agentName = "saved_agent"
timeConstraint = timedelta(seconds=30)

terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()
#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")

while True:
    startWorkTime = datetime.now()
    startWorkTime = datetime( year=startWorkTime.year, month=startWorkTime.month, day=startWorkTime.day,
                              hour = 4, minute=10, second=0, microsecond=0)
    startWorkTime = int( startWorkTime.timestamp() )

    endWorkTime = datetime.now()
    endWorkTime = datetime(year=endWorkTime.year, month=endWorkTime.month, day=endWorkTime.day,
                             hour=4, minute=15, second=0, microsecond=0)
    endWorkTime = int(endWorkTime.timestamp())

    while True:
        currentTime = datetime.now()
        currentTime = int(currentTime.timestamp())
        startDelta = startWorkTime - currentTime
        endDelta = endWorkTime - currentTime
        if startDelta > 0 or endDelta < 0:
            time.sleep(5)
        else:
            break



    #cycleTrain(saveDir, genPath, agentName)

    df = SymbolDataManager().getData(symbol, timeframe)
    ##########
    df = df.tail(10000)
    ##########
    preprocDf = preprocData(df)
    useAgent(preprocDf, hkFeatList, saveDir, genPath, agentName, timeConstraint)