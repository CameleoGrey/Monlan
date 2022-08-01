
from monlan.agents.DQNAgent import DQNAgent
from monlan.agents.CompositeAgent import CompositeAgent
from monlan.envs.CompositeEnv import CompositeEnv
#from monlan.feature_generators.FeatureScaler import FeatureScaler
from monlan.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from monlan.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from monlan.feature_generators.W2VDiffGenerator import W2VDiffGenerator
from monlan.datamanagement.SymbolDataManager import SymbolDataManager
from monlan.mods.VSASpread import VSASpread
from datetime import datetime
import matplotlib.pyplot as plt
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

symbol = "EURUSD_i"
timeframe = "H1"
priceFeatList = ["open", "close", "low", "high"]
#energyFeatList = ["vsa_spread"]
volumeFeatList = ["tick_volume"]

#terminal = MT5Terminal(login=99999999, server="Broker-MT5-Demo", password="password")
#dataUpdater = SymbolDataUpdater("../data/raw/")
dataManager = SymbolDataManager("../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager("../data/raw/").getData(symbol, timeframe)
modDf = df


priceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList, nDiffs=1, nPoints = 32, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=5, iter=100, min_count=0, sample=0.0, sg=0)
volumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList, nDiffs=1, nPoints = 32, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=5, iter=100, min_count=0, sample=0.0, sg=0)


priceDiffGenerator.setFitMode(True)
priceDiffGenerator = priceDiffGenerator.globalFit(df)
#priceDiffGenerator.checkReconstructionQuality(df)
priceDiffGenerator.saveGenerator("../models/w2vPriceDiffGen.pkl")

volumeDiffGenerator.setFitMode(True)
volumeDiffGenerator = volumeDiffGenerator.globalFit(df)
#volumeDiffGenerator.checkReconstructionQuality(df)
volumeDiffGenerator.saveGenerator("../models/w2vVolumeDiffGen.pkl")

openerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("../models/w2vPriceDiffGen.pkl")
buyerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("../models/w2vPriceDiffGen.pkl")
sellerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("../models/w2vPriceDiffGen.pkl")

openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("../models/w2vVolumeDiffGen.pkl")
buyerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("../models/w2vVolumeDiffGen.pkl")
sellerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("../models/w2vVolumeDiffGen.pkl")

openerCompositeGenerator = W2VCompositeGenerator( [openerPriceDiffGenerator,openerVolumeDiffGenerator], flatStack=False)
buyerCompositeGenerator = W2VCompositeGenerator( [buyerPriceDiffGenerator,buyerVolumeDiffGenerator], flatStack=False)
sellerCompositeGenerator = W2VCompositeGenerator( [sellerPriceDiffGenerator,sellerVolumeDiffGenerator], flatStack=False)

print("data updated")
print("start train")
################################
# train agent
################################
startTime = datetime.now()
print("start time: {}".format(startTime))
trainDf = modDf.tail(30000).head(25000)
trainEnv = CompositeEnv(trainDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                        stopType="const", takeType="const", stopPos=2, takePos=1, maxLoss=20000, maxTake=20000,
                        stoplossPuncts=20000, takeprofitPuncts=20000, riskPoints=110, riskLevels=5, parallelOpener=False,
                        renderFlag=True, renderDir="../models/", renderName="train_plot")
backTestDf = modDf.tail(35000).head(5000)
#backTestDf = modDf.head(50000).tail(3192).head(1192)
backTestEnv = CompositeEnv(backTestDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                           startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                           stopType="adaptive", takeType="adaptive", stopPos=1, takePos=1, maxLoss=20000, maxTake=20000,
                           stoplossPuncts=20000, takeprofitPuncts=20000, riskPoints=110, riskLevels=5, parallelOpener=False,
                           renderDir="../models/", renderName="back_plot")
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
#agent  = agent.load_agent("../models/", "best_composite")
#agent.agents["opener"].epsilon_min = 0.1
#agent.agents["buy"].epsilon_min = 0.1
#agent.agents["sell"].epsilon_min = 0.1
#agent  = agent.load_agent("../models/", "checkpoint_composite")
###################################
lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=20, nWarmUp=0,
                             uniformEps=False, synEps=False, plotScores=False,
                             saveBest=True, saveFreq=1, saveDir="../models/", saveName="best_composite")

endTime = datetime.now()
print("Training finished. Total time: {}".format(endTime - startTime))

###############################
# use agent
###############################
testDf = modDf.tail(30000).tail(5000)

openerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("../models/w2vPriceDiffGen.pkl")
buyerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("../models/w2vPriceDiffGen.pkl")
sellerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("../models/w2vPriceDiffGen.pkl")

openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("../models/w2vVolumeDiffGen.pkl")
buyerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("../models/w2vVolumeDiffGen.pkl")
sellerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("../models/w2vVolumeDiffGen.pkl")

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
                       startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                       stopType="adaptive", takeType="adaptive", stopPos=2, takePos=1, maxLoss=20000, maxTake=20000,
                       stoplossPuncts=20000, takeprofitPuncts=20000, riskPoints=110, riskLevels=5, parallelOpener=False,
                       renderDir="../models/", renderName="test_plot")
# get size of state and action from environment
openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent  = agent.load_agent("../models/", "best_composite", dropSupportModel=True)
print("start using agent")

dealsStatistics = agent.use_agent(testEnv)
sumRew = 0
cumulativeReward = []
for i in range(len(dealsStatistics)):
    sumRew += dealsStatistics[i]
    cumulativeReward.append(sumRew)
plt.plot( [x for x in range(len(cumulativeReward))], cumulativeReward )
plt.show()
