
"""
Trying to use multi scaler agent at demo
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
from avera.terminal.MT5Terminal import MT5Terminal
from avera.mods.HeikenAshiMod import HeikenAshiMod
from avera.mods.EnergyMod import EnergyMod
from avera.mods.VSASpread import VSASpread
import matplotlib.pyplot as plt
from datetime import datetime
from avera.feature_generators.CompositeGenerator import CompositeGenerator
from avera.utils.keras_utils import useCpu

def trainAgent(trainDf, backTestDf):

    openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
    buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
    sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")

    trainEnv = CompositeEnv(trainDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                            startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                            renderFlag=True)
    backTestEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                               startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                               renderFlag=True)
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
    # agent  = agent.load_agent("./", "best_composite")
    # agent  = agent.load_agent("./", "checkpoint_composite")
    lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=5, nWarmUp=0,
                    uniformEps=False, synEps=True, plotScores=True, saveBest=True, saveFreq=1)

    endTime = datetime.now()
    print("Training finished. Total time: {}".format(endTime - startTime))
    return lastSaveEp

def evaluateAgent(backTestDf):

    openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
    buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
    sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")

    testEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                           startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                           renderFlag=True)

    # get size of state and action from environment
    openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
    buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
    sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    agent = agent.load_agent("./", "best_composite")
    # agent  = agent.load_agent("./", "checkpoint_composite")
    print("start using agent")
    dealsStatistics = agent.use_agent(testEnv)
    sumRew = 0
    for i in range(len(dealsStatistics)):
        sumRew += dealsStatistics[i]
    return sumRew


#useCpu(nThreads=8, nCores=8)

startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]

terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

while True:
    dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
    df = SymbolDataManager().getData(symbol, timeframe)

    ###############
    historyMod = VSASpread()
    trainDf = historyMod.modHistory(df)
    historyMod = HeikenAshiMod()
    trainDf = historyMod.modHistory(trainDf)
    historyMod = EnergyMod()
    trainDf = historyMod.modHistory(trainDf, featList=["open", "close", "low", "high"])
    ###############

    priceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList, nDiffs=1, nPoints=32, flatStack=False, fitOnStep=False)
    priceDiffGenerator.setFitMode(True)
    priceDiffGenerator = priceDiffGenerator.globalFit(trainDf)
    priceDiffGenerator.saveGenerator("./MSDiffGen.pkl")

    trainDf = trainDf.tail(1032)
    backTestDf = trainDf.tail(1232).head(232)

    lastSaveEp = trainAgent(trainDf=trainDf, backTestDf=backTestDf)

    #####
    #break
    #####

    sumReward = evaluateAgent(backTestDf=backTestDf)
    if lastSaveEp > 0 and sumReward > 0:
        break
    else:
        print("Backtest failed. Retrain.")


###############################
# use agent at real environment
###############################

#openerPriceDiffGenerator = CompositeGenerator().loadGenerator("./CompositeMSDiffGen.pkl")
#buyerPriceDiffGenerator = CompositeGenerator().loadGenerator("./CompositeMSDiffGen.pkl")
#sellerPriceDiffGenerator = CompositeGenerator().loadGenerator("./CompositeMSDiffGen.pkl")

openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")

openerPriceDiffGenerator.setFitMode(False)
buyerPriceDiffGenerator.setFitMode(False)
sellerPriceDiffGenerator.setFitMode(False)

testEnv = RealCompositeEnv(symbol, timeframe, terminal, dataUpdater, dataManager,
                           openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                           startDeposit=300, lotSize=0.1, lotCoef=100000, renderFlag=True)

# get size of state and action from environment
openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent  = agent.load_agent("./", "best_composite", dropSupportModel=True)
#agent  = agent.load_agent("./", "checkpoint_composite")
print("start using agent")
agent.use_agent(testEnv)
