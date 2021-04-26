"""
ResNet + MultiScaler + extended rows
"""
from monlan.agents.DQNAgent import DQNAgent
from monlan.agents.CompositeAgent import CompositeAgent
from monlan.envs.CompositeEnv import CompositeEnv
from monlan.envs.RealCompositeEnv import RealCompositeEnv
from monlan.feature_generators.CompositeGenerator import CompositeGenerator
from monlan.feature_generators.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from monlan.datamanagement.SymbolDataManager import SymbolDataManager
from monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from monlan.terminal.MT5Terminal import MT5Terminal
from monlan.mods.HeikenAshiMod import HeikenAshiMod
from monlan.mods.EnergyMod import EnergyMod
from monlan.mods.VSASpread import VSASpread
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from monlan.utils.keras_utils import useCpu, reset_keras
import numpy as np

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
                               startDeposit=300, lotSize=0.1, lotCoef=100000, stopType="adaptive", takeType="adaptive",
                               stoplossPuncts=100, takeprofitPuncts=200, stopPos=2, takePos=1, maxLoss=40000,
                               maxTake=40000, riskPoints=110, riskLevels=5, renderFlag=False)

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

####################################################################################

symbol = "EURUSD_i"
timeframe = "M5"
hkFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]
saveDir = "../models/"
genName = "MSDiffGen"
genPath = saveDir + genName + ".pkl"
agentName = "best_composite"
timeConstraint = timedelta(days=5, hours=12, minutes=0)

terminal = MT5Terminal(login=50859400, server="Alpari-MT5-Demo", password="Q7P7Ly3Wi")
#terminal = MT5Terminal(login=99999999, server="Broker-MT5-Demo", password="password")
dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()

dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
while True:
    useAgent(symbol, timeframe, terminal, dataUpdater, dataManager,
             hkFeatList, saveDir, genPath, agentName, timeConstraint)