
################################################################
# Make an opener on-fly sample generator on a one symbol just
# for test. Using trained big supervised closer from the previous scripts
# to collect targets
################################################################

from src.monlan.modular_agents.StochasticCloser import StochasticCloser
from src.monlan.feature_generators.NoneGenerator import NoneGenerator
from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.feature_generators.OnFlySupervisedOpenerGenerator import OnFlySupervisedOpenerGenerator
from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from datetime import datetime
import os
import numpy as np
from src.monlan.utils.save_load import *
from tqdm import tqdm


start_time = datetime.now()
print("start time: {}".format(start_time))

symbol = "EURUSDrfd"
timeframe = "M5"
feat_list = ["open", "close", "low", "high", "tick_volume", "delta", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
data_updater = SymbolDataUpdater("../../data/raw/")
data_manager = SymbolDataManager("../../data/raw/")

#data_updater.full_update(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = data_manager.get_data(symbol, timeframe, normalize_names=True, normalize_date_time=True)
#df = df.tail(50000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True, add_raw_delta=True)
########


################################
# train agent
################################

base_feat_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
#base_feat_generator.fit_normalizing( mod_df )
stub_generator = NoneGenerator(feat_list, n_points=256, flat_stack=False)

opener_base_feat_generator = base_feat_generator
buyer_feat_generator = base_feat_generator
seller_base_feat_generator = base_feat_generator

#history_df = mod_df.tail(340000).head(300000)
full_histrory_size = len(mod_df)
back_size = 73000 # approximately one year on 5M timeframe
test_size = 73000 # approximately one year on 5M timeframe
train_size = full_histrory_size - (back_size + test_size)
#history_df = mod_df.tail(train_size + test_size).head( train_size )
####
# debug
history_df = mod_df.tail(train_size + test_size).head( 5000 )
####
composite_env = CompositeEnv(history_df, opener_base_feat_generator, buyer_feat_generator, seller_base_feat_generator,
                       start_deposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                       stop_type="const", take_type="const", stop_pos=2, take_pos=2, max_loss=20000, max_take=20000,
                       stoploss_puncts=2000, takeprofit_puncts=2000, risk_points=110, risk_levels=5, parallel_opener=False,
                       render_dir="../../data/pictures", render_name="test_plot", env_name="test_env", turn_off_spread=True)

trained_closer = load( os.path.join( "../../models", "best_val_resnet_closer.pkl" ) )
sample_generator = OnFlySupervisedOpenerGenerator(symbol=symbol, timeframe=timeframe, base_generator=base_feat_generator)
sample_generator.fit( history_df, composite_env, trained_closer,
                      norm_ema=0, smooth_ema=10,
                      show_plots=False )

save( sample_generator, os.path.join( "../../data/interim/opener_sample_generator.pkl" ) )