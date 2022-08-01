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
from datetime import datetime

startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "H1"
hkFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]

#terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager().getData(symbol, timeframe)

########
historyMod = VSASpread()
modDf = historyMod.modHistory(df)
historyMod = HeikenAshiMod()
modDf = historyMod.modHistory(modDf)
historyMod = EnergyMod()
modDf = historyMod.modHistory(modDf, featList=["open", "close", "low", "high"])
########

genList = []
for i in range(2):
    gen = MultiScalerDiffGenerator(featureList=hkFeatList, nDiffs=i+1, nPoints = 96, flatStack = False, fitOnStep = False)
    genList.append( gen )

compositeGenerator = CompositeGenerator( genList=genList, flatStack=False)
compositeGenerator.setFitMode(True)
compositeGenerator.globalFit(modDf)
compositeGenerator.saveGenerator("./CompositeMSDiffGen.pkl")

################################
# train agent
################################
openerPriceDiffGenerator = CompositeGenerator().loadGenerator("./CompositeMSDiffGen.pkl")
buyerPriceDiffGenerator = CompositeGenerator().loadGenerator("./CompositeMSDiffGen.pkl")
sellerPriceDiffGenerator = CompositeGenerator().loadGenerator("./CompositeMSDiffGen.pkl")

trainDf = modDf.tail(7200).head(5200)
trainEnv = CompositeEnv(trainDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001, renderFlag=True)
backTestDf = modDf.tail(9200).head(2200)
backTestEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001, renderFlag=True)
# get size of state and action from environment
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=2500, batch_size=100, train_start=200, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=0.9999, learning_rate=0.0001)
buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                          epsilon_decay=1.0, learning_rate=0.0001)
sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=1.0, learning_rate=0.0001)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
#agent  = agent.load_agent("./", "best_composite")
#agent  = agent.load_agent("./", "checkpoint_composite")
lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=25, nWarmUp=0,
                    uniformEps=False, synEps=True, plotScores=True, saveBest=True, saveFreq=1)

endTime = datetime.now()
print("Training finished. Total time: {}".format(endTime - startTime))


###############################
# use agent
###############################
testDf = modDf.tail(7200).tail(2200)
#testDf = modDf.tail(6032).head(5032)
#testDf = modDf.tail(6032).tail(1032)

openerPriceDiffGenerator = CompositeGenerator().loadGenerator("./CompositeMSDiffGen.pkl")
buyerPriceDiffGenerator = CompositeGenerator().loadGenerator("./CompositeMSDiffGen.pkl")
sellerPriceDiffGenerator = CompositeGenerator().loadGenerator("./CompositeMSDiffGen.pkl")

#openerPriceDiffGenerator.setFitMode(False)
#buyerPriceDiffGenerator.setFitMode(False)
#sellerPriceDiffGenerator.setFitMode(False)

testEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001, renderFlag=True)

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
