from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
from avera.envs.RealCompositeEnv import RealCompositeEnv
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from avera.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from avera.feature_generators.W2VDiffGenerator import W2VDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.terminal.MT5Terminal import MT5Terminal
from avera.mods.HeikenAshiMod import HeikenAshiMod


symbol = "EURUSD_i"
timeframe = "M15"
hkFeatList = ["hkopen", "hkclose", "low", "high"]
volumeFeatList = ["tick_volume"]

#terminal = MT5Terminal()
#dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager().getData(symbol, timeframe)

########
#df = df.tail(10000)
print("applying heiken ashi mod...")
enMod = HeikenAshiMod()
df = enMod.modHistory(df)
#enMod.checkQuality(df)
########


priceDiffGenerator = W2VDiffGenerator(featureList=hkFeatList, nDiffs=1, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=3, iter=100, min_count=0, sample=0.0, sg=0)
volumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList, nDiffs=1, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=3, iter=100, min_count=0, sample=0.0, sg=0)

priceDiffGenerator.setFitMode(True)
priceDiffGenerator = priceDiffGenerator.globalFit(df)
#priceDiffGenerator.checkReconstructionQuality(df)
priceDiffGenerator.saveGenerator("./w2vPriceDiffGen.pkl")

volumeDiffGenerator.setFitMode(True)
volumeDiffGenerator = volumeDiffGenerator.globalFit(df)
volumeDiffGenerator.saveGenerator("./w2vVolumeDiffGen.pkl")

################################
# train agent
################################
openerPriceDiffGenerator = W2VDiffGenerator(featureList=hkFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
buyerPriceDiffGenerator = W2VDiffGenerator(featureList=hkFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
sellerPriceDiffGenerator = W2VDiffGenerator(featureList=hkFeatList).loadGenerator("./w2vPriceDiffGen.pkl")

openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
buyerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
sellerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")

openerCompositeGenerator = W2VCompositeGenerator( [openerPriceDiffGenerator, openerVolumeDiffGenerator], flatStack=False)
buyerCompositeGenerator = W2VCompositeGenerator( [buyerPriceDiffGenerator, buyerVolumeDiffGenerator], flatStack=False)
sellerCompositeGenerator = W2VCompositeGenerator( [sellerPriceDiffGenerator, sellerVolumeDiffGenerator], flatStack=False)

#trainDf = df[:int(len(df)*0.9)]
trainDf = df.tail(2044).head(1022)
#trainDf = trainDf[:int(len(trainDf)*0.5)]
trainEnv = CompositeEnv(trainDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, renderFlag=True)
# get size of state and action from environment
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                        memorySize=500, batch_size=100, train_start=200, epsilon_min=0.05, epsilon=1, epsilon_decay=0.9994)
buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                        memorySize=1000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1, epsilon_decay=0.9999)
sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                        memorySize=1000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1, epsilon_decay=0.9999)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent.fit_agent(env=trainEnv, nEpisodes=15, nWarmUp = 0, uniformEps = False, plotScores=True, saveBest=True, saveFreq=2)

###############################
# use agent
###############################
#testDf = df.copy()#[int(len(df)*0.9):]
testDf = df.tail(2044).tail(1022) #check scaling at test env
#testDf = testDf[int(len(testDf)*0.5):]

openerPriceDiffGenerator = W2VDiffGenerator(featureList=hkFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
buyerPriceDiffGenerator = W2VDiffGenerator(featureList=hkFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
sellerPriceDiffGenerator = W2VDiffGenerator(featureList=hkFeatList).loadGenerator("./w2vPriceDiffGen.pkl")

openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
buyerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
sellerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")

openerPriceDiffGenerator.setFitMode(False)
buyerPriceDiffGenerator.setFitMode(False)
sellerPriceDiffGenerator.setFitMode(False)

openerVolumeDiffGenerator.setFitMode(False)
buyerVolumeDiffGenerator.setFitMode(False)
sellerVolumeDiffGenerator.setFitMode(False)

openerCompositeGenerator = W2VCompositeGenerator( [openerPriceDiffGenerator, openerVolumeDiffGenerator], flatStack=False)
buyerCompositeGenerator = W2VCompositeGenerator( [buyerPriceDiffGenerator, buyerVolumeDiffGenerator], flatStack=False)
sellerCompositeGenerator = W2VCompositeGenerator( [sellerPriceDiffGenerator, sellerVolumeDiffGenerator], flatStack=False)

testEnv = CompositeEnv(testDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, renderFlag=True)

#testEnv = RealCompositeEnv(symbol, timeframe, terminal, dataUpdater, dataManager,
#                           openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
#                           startDeposit=300, lotSize=0.1, lotCoef=100000, renderFlag=True)

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