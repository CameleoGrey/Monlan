
######################################################################
# Script for evaluating backtest performance of buyer and seller agents
# with random opener from different learning episodes inside first script.
# Running N times and collecting statistics to make scores.
######################################################################

import numpy as np

from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from src.monlan.utils.save_load import *
from datetime import datetime
from tqdm import tqdm
import os


start_time = datetime.now()
print("start time: {}".format(start_time))

symbol = "EURUSDrfd"
timeframe = "M10"
feat_list = ["open", "close", "low", "high", "tick_volume", "delta", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
data_updater = SymbolDataUpdater("../../../../../data/raw/")
data_manager = SymbolDataManager("../../../../../data/raw/")

#data_updater.full_update(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = data_manager.get_data(symbol, timeframe)
df = df.tail(380000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True, add_raw_delta=True)
########


################################
# train agent
################################
price_diff_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
opener_price_diff_generator = price_diff_generator
buyer_price_diff_generator = price_diff_generator
seller_price_diff_generator = price_diff_generator

#reward_transformer = RewardTransformer()
#reward_transformer.fit(df, spread_coef=0.00001, lot_coef=100000, lot_size=0.1, window_width=40)
#save(reward_transformer, path=os.path.join("../../models/reward_transformer.pkl"))
reward_transformer = load(os.path.join("../../../../../models/reward_transformer.pkl"))

test_df = mod_df.tail(380000).head(40000)
#back_test_df = mod_df.head(50000).tail(3192).head(1192)
test_env = CompositeEnv(test_df, opener_price_diff_generator, buyer_price_diff_generator, seller_price_diff_generator,
                        start_deposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                        stop_type="const", take_type="const", stop_pos=2, take_pos=2, max_loss=20000, max_take=20000,
                        stoploss_puncts=2000, takeprofit_puncts=2000, risk_points=110, risk_levels=5, parallel_opener=False,
                        render_dir="../../../../../data/pictures", render_name="test_plot", env_name="test_env", turn_off_spread=False)

scores = []
for i in tqdm(range( 2, 6 ), desc="Selected agent"):
    agent = load(os.path.join("../../../../../models/", "composite_agent_{}.pkl".format(i)))
    agent_scores = []
    for j in tqdm(range( 5 ), desc="Evaluation round"):
        score, deals_statistics = agent.use_agent(test_env, render_deals=False)
        mean_deal = np.mean( deals_statistics )
        agent_scores.append( mean_deal )
    agent_mean_score = np.mean( agent_scores )
    scores.append( agent_mean_score )

print( "Scores: {}".format( scores ) )
print( "Best score: {}".format( scores[ np.argmax(scores) ] ) )
print( "Best agent: {}".format( np.argmax(scores) ) )
print("done")