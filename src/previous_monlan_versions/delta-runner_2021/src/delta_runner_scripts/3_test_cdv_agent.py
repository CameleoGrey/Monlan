
from src.delta_runner.DeltaRunnerAgent import DeltaRunnerAgent
from src.monlan.modular_agents.CompositeAgent import CompositeAgent
from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.delta_runner.IdentFeatGen import IdentFeatGen
from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from src.monlan.utils.save_load import *
from datetime import datetime
import os
import numpy as np

startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "H1"
hkFeatList = ["open", "close", "low", "high", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = dataManager.getData(symbol, timeframe)
df = df.tail(400000)

########
mod_df = FeatGen_CDV().transform(df, period=32, verbose=True)
mod_df["cdv"] = mod_df["cdv"].apply(lambda x: np.sign(x) * np.log1p(np.abs(x)))

shifted = mod_df["cdv"].shift(1)
not_shifted = mod_df["cdv"]
mod_df["cdv"] = not_shifted - shifted
mod_df.dropna(inplace=True)
########

###############################
# use agent
###############################

model = load( os.path.join( "../../models", "cdv_predictor.pkl" ) )
base_threshold = load( os.path.join( "../../models", "base_threshold.pkl" ) )
scaler = load(os.path.join( "../../data", "scaler.pkl" ) )


priceDiffGenerator = IdentFeatGen(["cdv"], nPoints=128, flatStack=True)
openerPriceDiffGenerator = priceDiffGenerator
buyerPriceDiffGenerator = priceDiffGenerator
sellerPriceDiffGenerator = priceDiffGenerator

testDf = mod_df.tail(400000).tail(5000)
testEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                       startDeposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                       stopType="const", takeType="const", stopPos=2, takePos=2, maxLoss=20000, maxTake=20000,
                       stoplossPuncts=20000, takeprofitPuncts=20000, riskPoints=110, riskLevels=5, parallelOpener=False,
                       renderDir="../../data/pictures", renderName="test_plot", env_name="test_env", turn_off_spread=False)

# get size of state and action from environment


print("start using agent")
openerAgent = DeltaRunnerAgent( "opener", testEnv.observation_space["opener"], testEnv.action_space["opener"], model, scaler, base_threshold )
buyerAgent = DeltaRunnerAgent( "buyer", testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n, model, scaler, base_threshold)
sellerAgent = DeltaRunnerAgent( "seller", testEnv.observation_space["seller"], testEnv.action_space["seller"].n, model, scaler, base_threshold)

agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
print("start using agent")
score, dealsStatistics = agent.use_agent(testEnv, render_deals=True)

dealsStatistics = np.abs(dealsStatistics)
mean_deal = np.mean(dealsStatistics)
median_deal = np.median( dealsStatistics )
print(mean_deal)
print(median_deal)


print("done")