
from src.monlan.modular_agents.DQNAgent_pytorch import DQNAgent
from src.monlan.modular_agents.CompositeAgent import CompositeAgent
from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from src.monlan.utils.save_load import *
from datetime import datetime
import os


startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = dataManager.getData(symbol, timeframe)
df = df.tail(400000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True)
########


################################
# train agent
################################
openerPriceDiffGenerator = FeatGen_ScaledWindow(hkFeatList, nPoints=128, flatStack=False)
buyerPriceDiffGenerator = FeatGen_ScaledWindow(hkFeatList, nPoints=128, flatStack=False)
sellerPriceDiffGenerator = FeatGen_ScaledWindow(hkFeatList, nPoints=128, flatStack=False)

#reward_transformer = RewardTransformer()
#reward_transformer.fit(df, spread_coef=0.00001, lot_coef=100000, lot_size=0.1, window_width=40)
#save(reward_transformer, path=os.path.join("../../models/reward_transformer.pkl"))
reward_transformer = load( os.path.join("../../models/reward_transformer.pkl") )

trainDf = mod_df.tail(250000).head(200000)
trainEnv = CompositeEnv(trainDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                        stopType="const", takeType="const", stopPos=2, takePos=1, maxLoss=20000, maxTake=20000,
                        stoplossPuncts=2000, takeprofitPuncts=2000, riskPoints=110, riskLevels=5, parallelOpener=False,
                        renderFlag=True, renderDir="../../data/pictures", renderName="train_plot", env_name="train_env", turn_off_spread=True)

backTestDf = mod_df.tail(300000).head(50000)
backTestEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                           startDeposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                           stopType="const", takeType="const", stopPos=2, takePos=2, maxLoss=20000, maxTake=20000,
                           stoplossPuncts=2000, takeprofitPuncts=2000, riskPoints=110, riskLevels=5, parallelOpener=False,
                           renderDir="../../data/pictures", renderName="back_plot", env_name="back_env", turn_off_spread=False)

# get size of state and action from environment
openerAgent = DQNAgent( "opener", trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=40000, batch_size=1, train_start=30000, epsilon_min=0.99, epsilon=1, discount_factor=0.0,
                           epsilon_decay=0.9999, learning_rate=0.0001)
buyerAgent = DQNAgent( "buyer", trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=400000, batch_size=16, train_start=300000, epsilon_min=0.1, epsilon=1, discount_factor=0.999,
                          epsilon_decay=0.9999, learning_rate=0.0001)
sellerAgent = DQNAgent( "seller", trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=400000, batch_size=16, train_start=300000, epsilon_min=0.1, epsilon=1, discount_factor=0.999,
                           epsilon_decay=0.9999, learning_rate=0.0001)

agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent, reward_transformer=reward_transformer)
lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=100, nWarmUp=0,
                             uniformEps=False, synEps=False, plotScores=False,
                             saveBest=False, saveFreq=1, saveDir="../../models/", saveName="composite_agent")

endTime = datetime.now()
print("Training finished. Total time: {}".format(endTime - startTime))

###############################
# use agent
###############################

openerPriceDiffGenerator = FeatGen_ScaledWindow(hkFeatList, nPoints=256, flatStack=False)
buyerPriceDiffGenerator = FeatGen_ScaledWindow(hkFeatList, nPoints=48, flatStack=False)
sellerPriceDiffGenerator = FeatGen_ScaledWindow(hkFeatList, nPoints=48, flatStack=False)

testDf = mod_df.tail(250000).tail(50000)
#backTestDf = modDf.head(50000).tail(3192).head(1192)
testEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                           startDeposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                           stopType="const", takeType="const", stopPos=2, takePos=2, maxLoss=20000, maxTake=20000,
                           stoplossPuncts=200, takeprofitPuncts=100, riskPoints=110, riskLevels=5, parallelOpener=False,
                           renderDir="../../data/pictures", renderName="test_plot", env_name="test_env", turn_off_spread=False)

# get size of state and action from environment
agent = load( os.path.join("../../models/", "composite_agent_{}.pkl".format(1)))

print("start using agent")
score, dealsStatistics = agent.use_agent(testEnv)
print("done")