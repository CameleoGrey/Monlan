from avera.agents.DQNAgent import DQNAgent
from avera.envs.FeatureGenEnv import FeatureGenEnv
from avera.envs.RealEnv import RealEnv
from avera.terminal.MT5Terminal import MT5Terminal
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.ScalerGenerator import ScalerGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater

symbol = "TEST"
timeframe = "H1"
obsFeatList = ["open", "close"]

#terminal = MT5Terminal()
#dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager().getData(symbol, timeframe, normalizeDateTime=True, normalizeNames=True)
featureFactory = ScalerGenerator(featureList=obsFeatList, fitOnStep=False)
featureFactory.globalFit(df[:int(len(df)*0.9)])

print("data updated")
print("start train")
################################
# train agent
################################
trainDf = df[:int(len(df)*0.9)]
trainEnv = FeatureGenEnv(trainDf, featureFactory, startDeposit=10, lotSize=0.01, lotCoef=10, renderFlag=False)
# get size of state and action from environment
state_size = trainEnv.observation_space.shape[0]
action_size = trainEnv.action_space.n
agent = DQNAgent(state_size, action_size, epsilon_decay=0.999)
agent.fit_agent(env=trainEnv, nEpisodes=300, plotScores=True, saveFreq=10) #TODO: add save best only
agent.save_agent("./", "test_agent")
print("agent saved")

###############################
# use agent
###############################
testDF = df[int(len(df)*0.9):]
featureFactory = ScalerGenerator(featureList=obsFeatList, fitOnStep=True)
featureFactory.globalFit(df[:int(len(df)*0.9)])
testEnv = FeatureGenEnv(testDF, featureFactory, startDeposit=10, lotSize=0.01, lotCoef=10, renderFlag=True)
state_size = testEnv.observation_space.shape[0]
action_size = testEnv.action_space.n
print("loading agent")
agent = DQNAgent(state_size, action_size).load_agent(".", "test_agent")
agent.use_agent(testEnv)
print("start using agent")
#agent.use_agent(testEnv)
"""realEnv = RealEnv( symbol=symbol, timeframe=timeframe,
                   terminal=terminal, dataUpdater=dataUpdater, dataManager=dataManager, featureFactory=featureFactory,
                   obsFeatList=obsFeatList)
agent.use_agent(realEnv)"""