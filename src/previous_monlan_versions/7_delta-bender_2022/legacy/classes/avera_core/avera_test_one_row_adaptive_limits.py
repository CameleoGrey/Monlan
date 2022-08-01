"""
try avera approach
"""
from avera_core.AveraDQNAgent import DQNAgent
from avera_core.AveraDDQNAgent import DDQNAgent
from avera_core.AveraCompositeAgent import CompositeAgent
from avera_core.AveraCompositeEnv import AveraCompositeEnv
from avera_core.CloserFeatureGenerator import CloserFeatureGenerator
from avera_core.OpenerFeatureGenerator import OpenerFeatureGenerator
from avera.feature_generators.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.terminal.MT5Terminal import MT5Terminal
from avera.utils.save_load import *
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
hkFeatList = ["open", "close"]

#terminal = MT5Terminal()
#dataUpdater = SymbolDataUpdater("../data/raw/")
dataManager = SymbolDataManager("../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = dataManager.getData(symbol, timeframe)
#df = df.tail(50000)
modDf = df

"""averaCloserGen = CloserFeatureGenerator(featureList=["open", "close"],
                                            nFeatRows=1,
                                            nPoints=110,
                                            nLevels=5,
                                            flatStack=True,
                                            fitOnStep=True)
averaCloserGen = averaCloserGen.globalFit(modDf)
save("./closerFeatGen.pkl", averaCloseGen, verbose=True)"""

averaOpenerGen = OpenerFeatureGenerator( featureList=["open", "close"],
                                        nFeatRows=1,
                                        nPoints=110,
                                        nLevels=5,
                                        flatStack=True,
                                        fitOnStep=True)
averaCloserGen = load("./closerFeatGen.pkl")
averaOpenerGen.scaler = deepcopy( averaCloserGen.levelFeatsScaler )
save("./openerFeatGen.pkl", averaOpenerGen, verbose=True)
averaOpenerGen = load("./openerFeatGen.pkl")

################################
# train agent
################################
openerPriceDiffGenerator = load("./openerFeatGen.pkl")
buyerPriceDiffGenerator = load("./closerFeatGen.pkl")
sellerPriceDiffGenerator = load("./closerFeatGen.pkl")


trainDf = modDf.tail(100000).head(90000)
trainEnv = AveraCompositeEnv(trainDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                             startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                             stopType="adaptive", takeType="adaptive", stopPos=2, takePos=1, maxLoss=10000, maxTake=10000,
                             stoplossPuncts=20000, takeprofitPuncts=20000,
                             renderFlag=True, renderDir="./", renderName="train_plot")
backTestDf = modDf.tail(100000).tail(10000)
#backTestDf = modDf.head(50000).tail(3192).head(1192)
backTestEnv = AveraCompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                                startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                                stopType="adaptive", takeType="adaptive", stopPos=2, takePos=1, maxLoss=10000, maxTake=10000,
                                stoplossPuncts=20000, takeprofitPuncts=20000,
                                renderDir="./", renderName="back_plot")
# get size of state and action from environment
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=40000, batch_size=20, train_start=20000, epsilon_min=0.2, epsilon=1, discount_factor=0.99,
                           epsilon_decay=0.9999, learning_rate=0.0001)
buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=40000, batch_size=20, train_start=10000, epsilon_min=0.2, epsilon=1, discount_factor=0.99,
                          epsilon_decay=0.9999, learning_rate=0.0001)
sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=40000, batch_size=20, train_start=10000, epsilon_min=0.2, epsilon=1, discount_factor=0.99,
                           epsilon_decay=0.9999, learning_rate=0.0001)


agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
###################################
#agent  = agent.load_agent("./", "best_composite", dropSupportModel=True)
#agent.agents["opener"].epsilon_min = 0.1
#agent.agents["buy"].epsilon_min = 0.1
#agent.agents["sell"].epsilon_min = 0.1
#agent  = agent.load_agent("./", "checkpoint_composite")
###################################
lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=10, nWarmUp=0,
                                 uniformEps=False, synEps=False, plotScores=False, saveBest=True, saveFreq=1)

endTime = datetime.now()
print("Training finished. Total time: {}".format(endTime - startTime))

###############################
# use agent
###############################
from avera.utils.keras_utils import reset_keras
reset_keras()

#testDf = modDf.tail(12000).head(2000)
#testDf = modDf.tail(10000).head(8000)
testDf = modDf.tail(100000).tail(100000)
#testDf = modDf.tail(110000).head(10000) #back

openerPriceDiffGenerator = load("./openerFeatGen.pkl")
buyerPriceDiffGenerator = load("./closerFeatGen.pkl")
sellerPriceDiffGenerator = load("./closerFeatGen.pkl")

testEnv = AveraCompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                            startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                            stopType="adaptive", takeType="adaptive", stopPos=2, takePos=1, maxLoss=10000, maxTake=10000,
                            stoplossPuncts=20000, takeprofitPuncts=20000,
                            renderDir="./", renderName="test_plot")

# get size of state and action from environment
openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent  = agent.load_agent("./", "best_composite", dropSupportModel=True)
#agent  = agent.load_agent("./", "checkpoint_composite")
print("start using agent")

dealsStatistics = agent.use_agent(testEnv)
sumRew = 0
cumulativeReward = []
for i in range(len(dealsStatistics)):
    sumRew += dealsStatistics[i]
    cumulativeReward.append(sumRew)
plt.plot( [x for x in range(len(cumulativeReward))], cumulativeReward )
plt.show()
