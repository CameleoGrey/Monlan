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
from avera.feature_generators.W2VDiffGenerator import W2VDiffGenerator
from avera.feature_generators.RelativeDiffGenerator import RelativeDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
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
priceFeatList = ["open", "close", "low", "high"]
energyFeatList = ["enopen", "enclose", "enlow", "enhigh"]
hkFeatList = ["hkopen", "hkclose"]
vsaFeatList = ["vsa_spread"]
volumeFeatList = ["tick_volume"]

#terminal = MT5Terminal()
#dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager("../../data/raw/").getData(symbol, timeframe)
#df = df.tail(10000)

########
historyMod = VSASpread()
modDf = historyMod.modHistory(df)
historyMod = HeikenAshiMod()
modDf = historyMod.modHistory(modDf)
historyMod = EnergyMod()
modDf = historyMod.modHistory(modDf, featList=["open", "close", "low", "high"])
########
df = modDf

"""priceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList, nDiffs=1, nPoints = 84, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=84, window=10, iter=100, min_count=0, sample=0.0, sg=0)
energyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList, nDiffs=1, nPoints = 84, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=84, window=10, iter=100, min_count=0, sample=0.0, sg=0)
hkDiffGenerator = W2VDiffGenerator(featureList=hkFeatList, nDiffs=1, nPoints = 84, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=84, window=10, iter=100, min_count=0, sample=0.0, sg=0)
vsaDiffGenerator = W2VDiffGenerator(featureList=vsaFeatList, nDiffs=1, nPoints = 84, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=84, window=10, iter=100, min_count=0, sample=0.0, sg=0)
volumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList, nDiffs=1, nPoints = 84, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=84, window=10, iter=100, min_count=0, sample=0.0, sg=0)


priceDiffGenerator.setFitMode(True)
priceDiffGenerator = priceDiffGenerator.globalFit(df)
#priceDiffGenerator.checkReconstructionQuality(df)
priceDiffGenerator.saveGenerator("./w2vPriceDiffGen.pkl")

energyDiffGenerator.setFitMode(True)
energyDiffGenerator = energyDiffGenerator.globalFit(df)
energyDiffGenerator.saveGenerator("./w2vEnergyDiffGen.pkl")

hkDiffGenerator.setFitMode(True)
hkDiffGenerator = hkDiffGenerator.globalFit(df)
hkDiffGenerator.saveGenerator("./w2vHkDiffGen.pkl")

vsaDiffGenerator.setFitMode(True)
vsaDiffGenerator = vsaDiffGenerator.globalFit(df)
vsaDiffGenerator.saveGenerator("./w2vVSADiffGen.pkl")

volumeDiffGenerator.setFitMode(True)
volumeDiffGenerator = volumeDiffGenerator.globalFit(df)
volumeDiffGenerator.saveGenerator("./w2vVolumeDiffGen.pkl")


stubGen = MultiScalerDiffGenerator(featureList=priceFeatList, nDiffs=1, nPoints = 32, flatStack = False, fitOnStep = False)
stubGen.setFitMode(True)
stubGen = stubGen.globalFit(df)
stubGen.saveGenerator("./stubGen.pkl")"""

################################
# train agent
################################

openerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
openerEnergyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyDiffGen.pkl")
openerHkDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vHkDiffGen.pkl")
openerVSADiffGenerator = W2VDiffGenerator(featureList=vsaFeatList).loadGenerator("./w2vVSADiffGen.pkl")
openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
openerCompositeGenerator = W2VCompositeGenerator( [openerPriceDiffGenerator, openerEnergyDiffGenerator, openerHkDiffGenerator,
                                                   openerVSADiffGenerator, openerVolumeDiffGenerator], flatStack=False)

buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./stubGen.pkl")
sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./stubGen.pkl")

trainDf = modDf.tail(470000).tail(446500).head(423000)
trainEnv = CompositeEnv(trainDf, openerCompositeGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                        stoplossPuncts=100, takeprofitPuncts=200, renderFlag=True, renderDir="./", renderName="train_plot")
backTestDf = modDf.tail(4700000).head(23500)
#backTestDf = modDf.head(50000).tail(3192).head(1192)
backTestEnv = CompositeEnv(backTestDf, openerCompositeGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                           stoplossPuncts=100, takeprofitPuncts=200, renderFlag=True,
                           renderDir="./", renderName="back_plot")
# get size of state and action from environment.
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=600, batch_size=500, train_start=550, epsilon_min=0.2, epsilon=1,
                           epsilon_decay=0.9999, learning_rate=0.001)
buyerAgent = StubAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=600, batch_size=500, train_start=550, epsilon_min=0.2, epsilon=1,
                          epsilon_decay=0.9999, learning_rate=0.001)
sellerAgent = StubAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=600, batch_size=500, train_start=550, epsilon_min=0.2, epsilon=1,
                           epsilon_decay=0.9999, learning_rate=0.001)

agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
###################################
#agent  = agent.load_agent("./", "best_composite")
#agent  = agent.load_agent("./", "checkpoint_composite")
###################################
lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=1, nWarmUp=0,
                                 uniformEps=False, synEps=False, plotScores=False, saveBest=True, saveFreq=1)

endTime = datetime.now()
print("Training finished. Total time: {}".format(endTime - startTime))

###############################
# use agent
###############################
from avera.utils.keras_utils import reset_keras
reset_keras()

testDf = modDf.tail(470000)

openerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
openerEnergyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyDiffGen.pkl")
openerHkDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vHkDiffGen.pkl")
openerVSADiffGenerator = W2VDiffGenerator(featureList=vsaFeatList).loadGenerator("./w2vVSADiffGen.pkl")
openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
openerCompositeGenerator = W2VCompositeGenerator( [openerPriceDiffGenerator, openerEnergyDiffGenerator, openerHkDiffGenerator,
                                                   openerVSADiffGenerator, openerVolumeDiffGenerator], flatStack=False)

buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./stubGen.pkl")
sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=priceFeatList).loadGenerator("./stubGen.pkl")

#openerPriceDiffGenerator.setFitMode(False)
#buyerPriceDiffGenerator.setFitMode(False)
#sellerPriceDiffGenerator.setFitMode(False)

testEnv = CompositeEnv(testDf, openerCompositeGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
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