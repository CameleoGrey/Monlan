
################################################################
# Combining best buyer and seller agents with trained supervised
# opener agent into one composite agent and testing it on the
# forward data (last N candles that we didn't use through
# previous steps).
################################################################

from src.monlan.modular_agents.SupervisedOpener import SupervisedOpener
from src.monlan.modular_agents.AlwaysHoldAgent import AlwaysHoldAgent
from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from src.monlan.utils.save_load import *
from datetime import datetime
import os


start_time = datetime.now()
print("start time: {}".format(start_time))


symbol = "EURUSD_i"
timeframe = "M10"
feat_list = ["open", "close", "low", "high", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
data_updater = SymbolDataUpdater("../../data/raw/")
data_manager = SymbolDataManager("../../data/raw/")

#data_updater.full_update(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = data_manager.get_data(symbol, timeframe)
df = df.tail(380000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True)
########

###############################
# use agent
###############################

price_diff_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
opener_price_diff_generator = price_diff_generator
buyer_price_diff_generator = price_diff_generator
seller_price_diff_generator = price_diff_generator

test_df = mod_df.tail(380000).tail(40000)
#test_df = mod_df.tail(380000).head(40000)
#test_df = mod_df.tail(380000)
test_env = CompositeEnv(test_df, opener_price_diff_generator, buyer_price_diff_generator, seller_price_diff_generator,
                       start_deposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                       stop_type="const", take_type="const", stop_pos=2, take_pos=2, max_loss=20000, max_take=20000,
                       stoploss_puncts=2000, takeprofit_puncts=2000, risk_points=100, risk_levels=5, parallel_opener=False,
                       render_dir="../../data/pictures", render_name="test_plot", env_name="test_env", turn_off_spread=False)

# get size of state and action from environment
composite_agent = load( os.path.join("../../models/", "composite_agent_{}.pkl".format(1)))
supervised_opener = SupervisedOpener( "opener", test_env.observation_space["opener"], test_env.action_space["opener"].n )
so_model = load(os.path.join( "../../models", "best_val_supervised_opener_network.pkl" ))
#so_model = load(os.path.join( "../../models", "checkpoint_supervised_opener_network_{}.pkl".format(2) ))
supervised_opener.set_model( so_model, move_to_gpu=True )
composite_agent.agents["opener"] = supervised_opener

#####################
#composite_agent.agents["buyer"] = AlwaysHoldAgent("buyer", test_env.observation_space["buyer"])
#composite_agent.agents["seller"] = AlwaysHoldAgent("seller", test_env.observation_space["seller"])
#####################

print("start using agent")
score, deals_statistics = composite_agent.use_agent(test_env, render_deals=True)

import numpy as np
deals_statistics = np.abs(deals_statistics)
mean_deal = np.mean(deals_statistics)
median_deal = np.median( deals_statistics )
print(mean_deal)
print(median_deal)

print("done")