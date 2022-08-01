
################################################################
# Build CompositeAgent from the trained closer and opener, then
# test it on a data.
################################################################

from src.monlan.modular_agents.SupervisedOpener import SupervisedOpener
from src.monlan.modular_agents.CompositeAgent import CompositeAgent
from src.monlan.modular_agents.AlwaysHoldAgent import AlwaysHoldAgent
from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from src.monlan.utils.save_load import *
from src.monlan.terminal.MT5Terminal import MT5Terminal
from datetime import datetime
import os

from src.monlan.modular_agents.NaiveDistilledTCNMirrorOpener import NaiveDistilledTCNMirrorOpener


start_time = datetime.now()
print("start time: {}".format(start_time))


symbol = "EURUSDrfd"
timeframe = "M5"
feat_list = ["open", "close", "low", "high", "tick_volume", "delta", "cdv"]

data_updater = SymbolDataUpdater("../../data/raw/")
data_manager = SymbolDataManager("../../data/raw/")
df = data_manager.get_data(symbol, timeframe, normalize_names=True, normalize_date_time=True)
#df = df.tail(380000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True, add_raw_delta=True)
########

###############################
# use agent
###############################

price_diff_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
opener_price_diff_generator = price_diff_generator
buyer_price_diff_generator = price_diff_generator
seller_price_diff_generator = price_diff_generator


full_histrory_size = len(mod_df)
back_size = 73000 # approximately one year on 5M timeframe
test_size = 73000 # approximately one year on 5M timeframe
subsample_size = 73000 # approximately one year on 5M timeframe
train_size = full_histrory_size - (back_size + test_size)
#test_df = mod_df.tail(test_size).tail(subsample_size) # test subsample
#test_df = mod_df.head(back_size).tail(subsample_size) # back subsample
test_df = mod_df.tail(train_size + test_size).head(train_size).tail(subsample_size) # train subsample
test_env = CompositeEnv(test_df, opener_price_diff_generator, buyer_price_diff_generator, seller_price_diff_generator,
                       start_deposit=1000, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                       stop_type="const", take_type="const", stop_pos=2, take_pos=2, max_loss=20000, max_take=20000,
                       stoploss_puncts=2000, takeprofit_puncts=2000, risk_points=100, risk_levels=5, parallel_opener=False,
                       render_dir="../../data/pictures", render_name="best_val_resnet_agent", env_name="test_env", turn_off_spread=False)



supervised_opener = load( os.path.join( "../../models", "best_val_resnet_opener.pkl" ) )
mirror_buyer = load( os.path.join( "../../models", "best_val_resnet_closer.pkl" ) )
mirror_buyer.name = "buyer"
mirror_seller = load( os.path.join( "../../models", "best_val_resnet_closer.pkl" ) )
mirror_seller.name = "seller"
composite_agent = CompositeAgent( opener=supervised_opener,
                                  hold_buyer=mirror_buyer,
                                  hold_seller=mirror_seller,
                                  start_deposit=1000 )

print("start using agent")
score, deals_statistics = composite_agent.use_agent(test_env, render_deals=True)

import numpy as np
deals_statistics = deals_statistics
mean_deal = np.mean(deals_statistics)
median_deal = np.median( deals_statistics )
print(mean_deal)
print(median_deal)

print("done")