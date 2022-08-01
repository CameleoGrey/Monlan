
################################################################
# Make an opener on-fly sample generator on all symbols. Using
# trained big supervised closer from the previous scripts
# to collect targets for the future training of a big
# supervised opener.
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
from joblib import Parallel, delayed
from copy import deepcopy

def make_sample_generator(symbol, timeframe, feat_list):

    data_manager = SymbolDataManager("../../data/raw/")
    df = data_manager.get_data(symbol, timeframe, normalize_names=True, normalize_date_time=True)

    ########
    mod_df = FeatGen_CDV().transform(df, period=14, verbose=True, add_raw_delta=True)
    ########

    base_feat_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
    # base_feat_generator.fit_normalizing( mod_df )

    opener_base_feat_generator = base_feat_generator
    buyer_feat_generator = base_feat_generator
    seller_base_feat_generator = base_feat_generator

    full_histrory_size = len(mod_df)
    back_size = 73000 # approximately one year on 5M timeframe
    test_size = 73000 # approximately one year on 5M timeframe
    train_size = full_histrory_size - (back_size + test_size)
    train_df = mod_df.tail(train_size + test_size).head( train_size )

    if symbol in ["AUDUSDrfd", "USDCADrfd", "NZDUSDrfd", "USDCHFrfd", "EURUSDrfd", "GBPUSDrfd"]:
        spread_coef = 0.00001
        lot_coef = 100000
    elif symbol in ["USDJPYrfd"]:
        spread_coef = 0.001
        lot_coef = 1000
    else:
        raise Exception("Unknown symbol")

    composite_env = CompositeEnv(train_df, opener_base_feat_generator, buyer_feat_generator, seller_base_feat_generator,
                       start_deposit=1000, lot_size=0.1, lot_coef=lot_coef, spread=18, spread_coef=spread_coef,
                       stop_type="const", take_type="const", stop_pos=2, take_pos=2, max_loss=20000, max_take=20000,
                       stoploss_puncts=20000, takeprofit_puncts=20000, risk_points=110, risk_levels=5, parallel_opener=False,
                       render_dir="../../data/pictures", render_name="test_plot", env_name="test_env", turn_off_spread=True)

    sample_generator = OnFlySupervisedOpenerGenerator(symbol=symbol, timeframe=timeframe, base_generator=base_feat_generator)
    trained_closer = load(os.path.join("../../models", "best_val_resnet_closer.pkl"))
    sample_generator.fit( train_df, composite_env, trained_closer,
                      norm_ema=0, smooth_ema=10,
                      show_plots=False )

    save( sample_generator, os.path.join( "../../data/interim/opener_sample_generator_{}_{}.pkl".format(symbol, timeframe) ) )

if __name__ == "__main__":
    start_time = datetime.now()
    print("start time: {}".format(start_time))

    symbols = ["USDJPYrfd", "AUDUSDrfd", "USDCADrfd", "NZDUSDrfd", "USDCHFrfd", "EURUSDrfd", "GBPUSDrfd"]
    timeframe = "M5"
    feat_list = ["open", "close", "low", "high", "tick_volume", "delta", "cdv"]

    Parallel(n_jobs=1)(delayed(make_sample_generator)( symbol, deepcopy(timeframe), deepcopy(feat_list) ) for symbol in symbols)

