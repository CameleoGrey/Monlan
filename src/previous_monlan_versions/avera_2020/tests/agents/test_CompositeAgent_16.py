"""
multi scaler with backtest training and use at test env
"""
from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.agents.StubAgent import StubAgent
from avera.envs.CompositeEnv import CompositeEnv
from avera.envs.RealCompositeEnv import RealCompositeEnv
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from avera.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from avera.feature_generators.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from avera.feature_generators.RelativeDiffGenerator import RelativeDiffGenerator
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
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]

#terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = dataManager.getData(symbol, timeframe)
df = df.tail(10000)

########
historyMod = VSASpread()
modDf = historyMod.modHistory(df)
historyMod = HeikenAshiMod()
modDf = historyMod.modHistory(modDf)
historyMod = EnergyMod()
modDf = historyMod.modHistory(modDf, featList=["open", "close", "low", "high"])
########

priceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList, nDiffs=1, nPoints = 128, flatStack = False, fitOnStep = False)
priceDiffGenerator.setFitMode(True)
priceDiffGenerator = priceDiffGenerator.globalFit(modDf)
priceDiffGenerator.saveGenerator("./MSDiffGen.pkl")

#priceDiffGenerator = RelativeDiffGenerator(featureList=hkFeatList, nDiffs=1, nPoints = 512, flatStack = False, fitOnStep = False)
#priceDiffGenerator.setFitMode(True)
#priceDiffGenerator = priceDiffGenerator.globalFit(modDf)
#priceDiffGenerator.saveGenerator("./RDiffGen.pkl")

################################
# train agent
################################
openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")

trainDf = modDf.tail(470000).tail(120000).tail(100000)
trainEnv = CompositeEnv(trainDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=20, spreadCoef=0.00001,
                        stoplossPuncts=100, takeprofitPuncts=200, renderFlag=True, renderDir="./", renderName="train_plot")
backTestDf = modDf.tail(4700000).tail(120000).head(20000)
#backTestDf = modDf.head(50000).tail(3192).head(1192)
backTestEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=20, spreadCoef=0.00001,
                           stoplossPuncts=100, takeprofitPuncts=200, renderFlag=True,
                           renderDir="./", renderName="back_plot")
# get size of state and action from environment.
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                       memorySize=5000, batch_size=200, train_start=1200, epsilon_min=0.66, epsilon=1,
                       epsilon_decay=0.999, learning_rate=0.001, discount_factor=0.0)
buyerAgent = StubAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                       memorySize=10, batch_size=1, train_start=2, epsilon_min=0.01, epsilon=1,
                       epsilon_decay=0.999, learning_rate=0.0001)
sellerAgent = StubAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                        memorySize=10, batch_size=1, train_start=2, epsilon_min=0.01, epsilon=1,
                        epsilon_decay=0.999, learning_rate=0.0001)

agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
###################################
#agent  = agent.load_agent("./", "best_composite")
#agent  = agent.load_agent("./", "checkpoint_composite")
###################################
lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=30, nWarmUp=1,
                                 uniformEps=False, synEps=False, plotScores=False, saveBest=True, saveFreq=1)

endTime = datetime.now()
print("Training finished. Total time: {}".format(endTime - startTime))

###############################
# use agent
###############################
from avera.utils.keras_utils import reset_keras
reset_keras()

testDf = modDf.tail(470000)

openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")

#openerPriceDiffGenerator.setFitMode(False)
#buyerPriceDiffGenerator.setFitMode(False)
#sellerPriceDiffGenerator.setFitMode(False)

testEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=20, spreadCoef=0.00001,
                       stoplossPuncts=100, takeprofitPuncts=200, renderFlag=True,
                       renderDir="./", renderName="test_plot")

# get size of state and action from environment3
openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
buyerAgent = StubAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
sellerAgent = StubAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
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
