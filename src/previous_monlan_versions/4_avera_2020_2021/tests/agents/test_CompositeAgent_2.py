from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
from avera.envs.RealEnv import RealEnv
from avera.terminal.MT5Terminal import MT5Terminal
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.ScalerGenerator import ScalerGenerator
from avera.feature_generators.DiffGenerator import DiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater

symbol = "EURUSD_i"
timeframe = "H1"
obsFeatList = ["open", "close"]

#terminal = MT5Terminal()
#dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager().getData(symbol, timeframe)
df = df.tail(1000)
openerFeatureFactory = DiffGenerator(featureList=obsFeatList, nDiffs=1, nPoints=10, flatStack=True, fitOnStep=False)
buyerFeatureFactory = DiffGenerator(featureList=obsFeatList, nDiffs=1, nPoints=10, flatStack=True, fitOnStep=False)
sellerFatureFactory = DiffGenerator(featureList=obsFeatList, nDiffs=1, nPoints=10, flatStack=True, fitOnStep=False)
openerFeatureFactory.globalFit(df[:int(len(df)*0.9)])
buyerFeatureFactory.globalFit(df[:int(len(df)*0.9)])
sellerFatureFactory.globalFit(df[:int(len(df)*0.9)])

print("data updated")
print("start train")
################################
# train agent
################################
trainDf = df[:int(len(df)*0.9)]
trainEnv = CompositeEnv(trainDf, openerFeatureFactory, buyerFeatureFactory, sellerFatureFactory,
                        startDeposit=300, lotSize=0.01, lotCoef=100000, renderFlag=True)
# get size of state and action from environment
openerAgent = DQNAgent(trainEnv.observation_space["opener"][0], trainEnv.action_space["opener"].n,
                        memorySize=2000, batch_size=1000, train_start=1200, epsilon_decay=0.999)
buyerAgent = DQNAgent(trainEnv.observation_space["buyer"][0], trainEnv.action_space["buyer"].n,
                        memorySize=2000, batch_size=1000, train_start=1200, epsilon_decay=0.999)
sellerAgent = DQNAgent(trainEnv.observation_space["seller"][0], trainEnv.action_space["seller"].n,
                        memorySize=2000, batch_size=1000, train_start=1200, epsilon_decay=0.999)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent.fit_agent(env=trainEnv, nEpisodes=200, plotScores=True, saveFreq=10) #TODO: add save best only
agent.save_agent("./", "test_composite_agent")
print("agent saved")

###############################
# use agent
###############################
"""testDF = df[int(len(df)*0.9):]
featureFactory = ScalerGenerator(featureList=obsFeatList, fitOnStep=True)
featureFactory.globalFit(df[:int(len(df)*0.9)])
testEnv = FeatureGenEnv(testDF, featureFactory, startDeposit=10, lotSize=0.01, lotCoef=10, renderFlag=True)
state_size = testEnv.observation_space.shape[0]
action_size = testEnv.action_space.n
print("loading agent")
agent = DQNAgent(state_size, action_size).load_agent(".", "test_agent")
print("start using agent")
agent.use_agent(testEnv)"""
"""realEnv = RealEnv( symbol=symbol, timeframe=timeframe,
                   terminal=terminal, dataUpdater=dataUpdater, dataManager=dataManager, featureFactory=featureFactory,
                   obsFeatList=obsFeatList)
agent.use_agent(realEnv)"""