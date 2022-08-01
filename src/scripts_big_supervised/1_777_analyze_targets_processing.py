
################################################################
# Looking on a subsample of targets to check their quality.
################################################################

from src.monlan.modular_agents.StochasticCloser import StochasticCloser
from src.monlan.feature_generators.NoneGenerator import NoneGenerator
from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.feature_generators.OnFlySupervisedSampleGenerator import OnFlySupervisedSampleGenerator
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
timeframe = "M10"
feat_list = ["open", "close", "low", "high", "tick_volume", "delta", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
data_updater = SymbolDataUpdater("../../data/raw/")
data_manager = SymbolDataManager("../../data/raw/")

#data_updater.full_update(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = data_manager.get_data(symbol, timeframe)
df = df.tail(38000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True, add_raw_delta=True)
########


################################
# train agent
################################
buyer_agent = StochasticCloser()

base_feat_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
#base_feat_generator.fit_normalizing( mod_df )

stub_generator = NoneGenerator(feat_list, n_points=256, flat_stack=False)

opener_base_feat_generator = stub_generator
buyer_feat_generator = stub_generator
seller_base_feat_generator = stub_generator

history_df = mod_df.tail(2256)
composite_env = CompositeEnv(history_df, opener_base_feat_generator, buyer_feat_generator, seller_base_feat_generator,
                       start_deposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                       stop_type="const", take_type="const", stop_pos=2, take_pos=2, max_loss=20000, max_take=20000,
                       stoploss_puncts=2000, takeprofit_puncts=2000, risk_points=110, risk_levels=5, parallel_opener=False,
                       render_dir="../../data/pictures", render_name="test_plot", env_name="test_env", turn_off_spread=True)

sample_generator = OnFlySupervisedSampleGenerator(symbol=symbol, timeframe=timeframe, base_generator=base_feat_generator)
sample_generator.fit( history_df, composite_env,
                      norm_ema=0, smooth_ema=10,
                      start_hold_proba=0.9, end_hold_proba=0.95, n_estimates=100,
                      show_plots=False )
sample_generator.plot_ohlc_targets(candle_count=1000)

#save( sample_generator, os.path.join( "../../data/interim/tmp_trash.pkl" ) )