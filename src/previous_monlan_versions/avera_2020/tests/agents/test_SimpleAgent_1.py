from avera.agents.DQNAgent import DQNAgent
from avera.envs.SimpleEnv import SimpleEnv
from avera.envs.RealEnv import RealEnv
from avera.terminal.MT5Terminal import MT5Terminal
from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater

symbol = "EURUSD_i"
timeframe = "M5"
obsFeatList = ["open", "close"]

terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater()
dataManager = SymbolDataManager()
featureFactory = FeatureScaler()

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
df = SymbolDataManager().getData(symbol, timeframe)
featureScaler = FeatureScaler()
df = featureScaler.extractFeature(df, featList=obsFeatList)

print("data updated")
print("start train")
################################
# train agent
################################
trainDf = df[:int(len(df)*0.9)]
trainEnv = SimpleEnv(trainDf, obsFeatList=obsFeatList, renderFlag=False)
# get size of state and action from environment
state_size = trainEnv.observation_space.shape[0]
action_size = trainEnv.action_space.n
agent = DQNAgent(state_size, action_size, epsilon_decay=0.999)
agent.fit_agent(env=trainEnv, nEpisodes=1, plotScores=True, saveFreq=1) #TODO: add save best only
agent.save_agent("./", "test_agent")
print("agent saved")

###############################
# use agent
###############################
testDF = df[int(len(df)*0.98):]
testEnv = SimpleEnv(testDF, obsFeatList=obsFeatList, renderFlag=True)
state_size = testEnv.observation_space.shape[0]
action_size = testEnv.action_space.n
print("loading agent")
agent = DQNAgent(state_size, action_size).load_agent(".", "test_agent")
print("start using agent")
#agent.use_agent(testEnv)
realEnv = RealEnv( symbol=symbol, timeframe=timeframe,
                   terminal=terminal, dataUpdater=dataUpdater, dataManager=dataManager, featureFactory=featureFactory,
                   obsFeatList=obsFeatList)
agent.use_agent(realEnv)