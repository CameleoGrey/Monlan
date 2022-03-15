import numpy as np

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
priceDiffGenerator = FeatGen_ScaledWindow(hkFeatList, nPoints=256, flatStack=False)
openerPriceDiffGenerator = priceDiffGenerator
buyerPriceDiffGenerator = priceDiffGenerator
sellerPriceDiffGenerator = priceDiffGenerator

#reward_transformer = RewardTransformer()
#reward_transformer.fit(df, spread_coef=0.00001, lot_coef=100000, lot_size=0.1, window_width=40)
#save(reward_transformer, path=os.path.join("../../models/reward_transformer.pkl"))
reward_transformer = load( os.path.join("../../models/reward_transformer.pkl") )

testDf = mod_df.tail(140000).head(20000)
#backTestDf = modDf.head(50000).tail(3192).head(1192)
testEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                       startDeposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                       stopType="const", takeType="const", stopPos=2, takePos=2, maxLoss=20000, maxTake=20000,
                       stoplossPuncts=200, takeprofitPuncts=100, riskPoints=110, riskLevels=5, parallelOpener=False,
                       renderDir="../../data/pictures", renderName="test_plot", env_name="test_env", turn_off_spread=False)

scores = []
for i in range( 10 ):
    agent = load( os.path.join("../../models/", "composite_agent_{}.pkl".format(i)))
    agent_scores = []
    for j in range( 5 ):
        score, dealsStatistics = agent.use_agent(testEnv, render_deals=False)
        median_deal = np.median( dealsStatistics )
        agent_scores.append( median_deal )
    agent_mean_score = np.mean( agent_scores )
    scores.append( agent_mean_score )

print( "Scores: {}".format( scores ) )
print( "Best score: {}".format( scores[ np.argmax(scores) ] ) )
print( "Best agent: {}".format( np.argmax(scores) ) )
print("done")