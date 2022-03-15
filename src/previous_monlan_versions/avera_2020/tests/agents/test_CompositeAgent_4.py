from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
from avera.envs.RealEnv import RealEnv
from avera.terminal.MT5Terminal import MT5Terminal
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater

symbol = "TEST"
timeframe = "H1"
obsFeatList = ["open", "close"]

#terminal = MT5Terminal()
#dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager().getData(symbol, timeframe, normalizeNames=True, normalizeDateTime=True)
featureGenerator = W2VScaleGenerator(featureList=obsFeatList, nPoints = 40, flatStack = False, fitOnStep = False,
                 nIntervals = 1000, w2vSize=100, window=20, iter=100, min_count=0, sample=0.0, sg=0)
featureGenerator = featureGenerator.globalFit(df)
featureGenerator.saveGenerator("./w2vGen.pkl")
openerFeatureFactory = W2VScaleGenerator(featureList=obsFeatList).loadGenerator("./w2vGen.pkl")
buyerFeatureFactory = W2VScaleGenerator(featureList=obsFeatList).loadGenerator("./w2vGen.pkl")
sellerFatureFactory = W2VScaleGenerator(featureList=obsFeatList).loadGenerator("./w2vGen.pkl")

print("data updated")
print("start train")
################################
# train agent
################################
trainDf = df[:int(len(df)*0.6)]
trainEnv = CompositeEnv(trainDf, openerFeatureFactory, buyerFeatureFactory, sellerFatureFactory,
                        startDeposit=10, lotSize=0.01, lotCoef=10, renderFlag=True)
# get size of state and action from environment
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                        memorySize=1000, batch_size=100, train_start=200, epsilon_decay=0.999)
buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                        memorySize=2000, batch_size=200, train_start=300, epsilon_decay=0.999)
sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                        memorySize=2000, batch_size=200, train_start=300, epsilon_decay=0.999)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent.fit_agent(env=trainEnv, nEpisodes=30, plotScores=True, saveBest=True, saveFreq=2)

###############################
# use agent
###############################
testDF = df[int(len(df)*0.6):]
testEnv = CompositeEnv(testDF, openerFeatureFactory, buyerFeatureFactory, sellerFatureFactory,
                        startDeposit=10, lotSize=0.01, lotCoef=10, renderFlag=True)

print("loading agent")
agent  = CompositeAgent(None, None, None).load_agent("./", "best_composite")
print("start using agent")
agent.use_agent(testEnv)
"""realEnv = RealEnv( symbol=symbol, timeframe=timeframe,
                   terminal=terminal, dataUpdater=dataUpdater, dataManager=dataManager, featureFactory=featureFactory,
                   obsFeatList=obsFeatList)
agent.use_agent(realEnv)"""