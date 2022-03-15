"""
w2v with backtest
"""

from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from avera.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from avera.feature_generators.W2VDiffGenerator import W2VDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.mods.VSASpread import VSASpread
from datetime import datetime
import matplotlib.pyplot as plt

symbol = "EURUSD_i"
timeframe = "H1"
priceFeatList = ["open", "close", "low", "high"]
#energyFeatList = ["vsa_spread"]
volumeFeatList = ["tick_volume"]

#terminal = MT5Terminal()
#dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager("../../data/raw/").getData(symbol, timeframe)


priceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList, nDiffs=1, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=3, iter=100, min_count=0, sample=0.0, sg=0)
volumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList, nDiffs=1, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=3, iter=100, min_count=0, sample=0.0, sg=0)


priceDiffGenerator.setFitMode(True)
priceDiffGenerator = priceDiffGenerator.globalFit(df)
priceDiffGenerator.saveGenerator("./w2vPriceDiffGen.pkl")

volumeDiffGenerator.setFitMode(True)
volumeDiffGenerator = volumeDiffGenerator.globalFit(df)
volumeDiffGenerator.saveGenerator("./w2vVolumeDiffGen.pkl")

openerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
buyerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
sellerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")

openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
buyerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
sellerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")

openerCompositeGenerator = W2VCompositeGenerator( [openerPriceDiffGenerator,openerVolumeDiffGenerator], flatStack=False)
buyerCompositeGenerator = W2VCompositeGenerator( [buyerPriceDiffGenerator,buyerVolumeDiffGenerator], flatStack=False)
sellerCompositeGenerator = W2VCompositeGenerator( [sellerPriceDiffGenerator,sellerVolumeDiffGenerator], flatStack=False)

print("data updated")
print("start train")
################################
# train agent
################################
trainDf = df.tail(6200).head(5200)
trainEnv = CompositeEnv(trainDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001, renderFlag=True)
backTestDf = df.tail(7200).head(1200)
backTestEnv = CompositeEnv(backTestDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001, renderFlag=True)
# get size of state and action from environment
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                        memorySize=2500, batch_size=100, train_start=200, epsilon_min=0.05, epsilon=1, epsilon_decay=0.9999, learning_rate=0.0002)
buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                        memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1, epsilon_decay=1.0, learning_rate=0.0002)
sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                        memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1, epsilon_decay=1.0, learning_rate=0.0002)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
startTime = datetime.now()
print("start time: {}".format(startTime))
lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=10, nWarmUp=0,
                    uniformEps=False, synEps=True, plotScores=True, saveBest=True, saveFreq=1)
endTime = datetime.now()
print("Training finished. Total time: {}".format(endTime - startTime))

###############################
# use agent
###############################
testDf = df.tail(6200).tail(1200)

openerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
buyerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
sellerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")

openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
buyerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
sellerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")

#openerPriceDiffGenerator.setFitMode(False)
#buyerPriceDiffGenerator.setFitMode(False)
#sellerPriceDiffGenerator.setFitMode(False)

#openerVolumeDiffGenerator.setFitMode(False)
#buyerVolumeDiffGenerator.setFitMode(False)
#sellerVolumeDiffGenerator.setFitMode(False)

openerCompositeGenerator = W2VCompositeGenerator( [openerPriceDiffGenerator,openerVolumeDiffGenerator], flatStack=False)
buyerCompositeGenerator = W2VCompositeGenerator( [buyerPriceDiffGenerator,buyerVolumeDiffGenerator], flatStack=False)
sellerCompositeGenerator = W2VCompositeGenerator( [sellerPriceDiffGenerator,sellerVolumeDiffGenerator], flatStack=False)

testEnv = CompositeEnv(testDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, renderFlag=True)
# get size of state and action from environment
openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent  = agent.load_agent("./", "best_composite", dropSupportModel=True)
print("start using agent")

dealsStatistics = agent.use_agent(testEnv)
sumRew = 0
cumulativeReward = []
for i in range(len(dealsStatistics)):
    sumRew += dealsStatistics[i]
    cumulativeReward.append(sumRew)
plt.plot( [x for x in range(len(cumulativeReward))], cumulativeReward )
plt.show()
