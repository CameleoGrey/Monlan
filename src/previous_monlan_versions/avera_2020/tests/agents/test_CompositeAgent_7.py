from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from avera.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from avera.feature_generators.W2VDiffGenerator import W2VDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.mods.EnergyMod import EnergyMod

symbol = "EURUSD_i"
timeframe = "H1"
energyFeatList = ["enopen", "enclose", "enlow", "enhigh"]
volumeFeatList = ["tick_volume"]

#terminal = MT5Terminal()
#dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager().getData(symbol, timeframe)

########
#df = df.tail(1000)
print("applying energy mod...")
enMod = EnergyMod()
df = enMod.modHistory(df, ["open", "close", "low", "high"])
#enMod.checkQuality(df)
########

"""energyScaleGenerator = W2VScaleGenerator(featureList=energyFeatList, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=20, iter=100, min_count=0, sample=0.0, sg=0)
energyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList, nDiffs=1, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=20, iter=100, min_count=0, sample=0.0, sg=0)"""
"""volumeScaleGenerator = W2VScaleGenerator(featureList=volumeFeatList, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=20, iter=100, min_count=0, sample=0.0, sg=0)
volumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList, nDiffs=1, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=20, iter=100, min_count=0, sample=0.0, sg=0)"""

"""energyScaleGenerator.setFitMode(True)
energyScaleGenerator = energyScaleGenerator.globalFit(df)
#priceScaleGenerator.checkReconstructionQuality(df.head(10000))
#priceScaleGenerator.checkReconstructionQuality(df.tail(10000))
energyScaleGenerator.saveGenerator("./w2vEnergyScaleGen.pkl")

energyDiffGenerator.setFitMode(True)
energyDiffGenerator = energyDiffGenerator.globalFit(df)
energyDiffGenerator.saveGenerator("./w2vEnergyDiffGen.pkl")"""


"""openerEnergyScalerGenerator = W2VScaleGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyScaleGen.pkl")
buyerEnergyScalerGenerator = W2VScaleGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyScaleGen.pkl")
sellerEnergyScalerGenerator = W2VScaleGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyScaleGen.pkl")

openerEnergyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyDiffGen.pkl")
buyerEnergyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyDiffGen.pkl")
sellerEnergyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyDiffGen.pkl")

openerCompositeGenerator = W2VCompositeGenerator( [openerEnergyScalerGenerator, openerEnergyDiffGenerator], flatStack=False)
buyerCompositeGenerator = W2VCompositeGenerator( [buyerEnergyScalerGenerator, buyerEnergyDiffGenerator], flatStack=False)
sellerCompositeGenerator = W2VCompositeGenerator( [sellerEnergyScalerGenerator, sellerEnergyDiffGenerator], flatStack=False)

print("data updated")
print("start train")
################################
# train agent
################################
#trainDf = df[:int(len(df)*0.9)]
trainDf = df.tail(1000)
trainDf = trainDf[:int(len(trainDf)*0.8)]
trainEnv = CompositeEnv(trainDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=300, lotSize=0.01, lotCoef=100000, renderFlag=True)
# get size of state and action from environment
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                        memorySize=500, batch_size=100, train_start=200, epsilon_min=0.01, epsilon=1, epsilon_decay=0.9993)
buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                        memorySize=1000, batch_size=200, train_start=300, epsilon_min=0.01, epsilon=1, epsilon_decay=0.9999)
sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                        memorySize=1000, batch_size=200, train_start=300, epsilon_min=0.01, epsilon=1, epsilon_decay=0.9999)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent.fit_agent(env=trainEnv, nEpisodes=20, nWarmUp = 1, uniformEps = False, plotScores=True, saveBest=True, saveFreq=2)"""

###############################
# use agent
###############################
#testDf = df.copy()#[int(len(df)*0.9):]
testDf = df.tail(222) #check scaling at test env
#testDf = testDf[int(len(testDf)*0.8):]

openerEnergyScalerGenerator = W2VScaleGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyScaleGen.pkl")
buyerEnergyScalerGenerator = W2VScaleGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyScaleGen.pkl")
sellerEnergyScalerGenerator = W2VScaleGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyScaleGen.pkl")

openerEnergyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyDiffGen.pkl")
buyerEnergyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyDiffGen.pkl")
sellerEnergyDiffGenerator = W2VDiffGenerator(featureList=energyFeatList).loadGenerator("./w2vEnergyDiffGen.pkl")

"""openerPriceScalerGenerator.setFitMode(False)
buyerPriceScalerGenerator.setFitMode(False)
sellerPriceScalerGenerator.setFitMode(False)
openerPriceDiffGenerator.setFitMode(False)
buyerPriceDiffGenerator.setFitMode(False)
sellerPriceDiffGenerator.setFitMode(False)"""

openerCompositeGenerator = W2VCompositeGenerator( [openerEnergyScalerGenerator, openerEnergyDiffGenerator], flatStack=False)
buyerCompositeGenerator = W2VCompositeGenerator( [buyerEnergyScalerGenerator, buyerEnergyDiffGenerator], flatStack=False)
sellerCompositeGenerator = W2VCompositeGenerator( [sellerEnergyScalerGenerator, sellerEnergyDiffGenerator], flatStack=False)

testEnv = CompositeEnv(testDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=300, lotSize=0.01, lotCoef=100000, renderFlag=True)
# get size of state and action from environment
openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent  = agent.load_agent("./", "best_composite")
print("start using agent")
agent.use_agent(testEnv)
"""realEnv = RealEnv( symbol=symbol, timeframe=timeframe,
                   terminal=terminal, dataUpdater=dataUpdater, dataManager=dataManager, featureFactory=featureFactory,
                   obsFeatList=obsFeatList)
agent.use_agent(realEnv)"""