
################################################################
# The main idea of Big Supervised Monlan is generating stochastic
# value estimation for each history step. Then there's training a model that
# predicts values for buy and sell. Based on this model we're building
# mirror closer: if it plays the role of a buyer, then its hold action value estimates
# as buyer value prediction and close action value estimates as sell value prediction.
# (A mirror closer in the role of buyer holds position while it is more profitable
# than close and switch to the seller.)
# Then a trained mirror closer is using for generating real performance estimation
# for an opener in each history point of a training data. Then we add
# hold estimates to these targets and train an opener. It is important because
# no one can trade profitable without the ability to wait a good moment to open a deal.
# After that we combine the trained opener and the mirror closer into one
# CompositeAgent that is testing on a test data in the last script.
# This idea came from experiments with pure RL and distilled agents.
# This method combined with on-fly training sample generation makes
# it possible to perform relatively fast and effective training of an agent
# on a big data (millions of history samples) on a modern
# laptop (64 Gb DDR4 RAM, i7-11800H, RTX3060 6Gb).
# To complete the whole pipeline from scratch one needs ~5-7 days.
################################################################

################################################################
# Making on-fly sample generators for a list of symbols.
# They will be used to make meta sample generator that is needful
# to train big supervised closer (next scripts).
# Without on-fly generator training need a huge amount of RAM memory.
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
from joblib import Parallel, delayed
from copy import deepcopy

def make_sample_generator(symbol, timeframe, feat_list):

    data_manager = SymbolDataManager("../../data/raw/")
    df = data_manager.get_data(symbol, timeframe, normalize_names=True, normalize_date_time=True)

    ########
    mod_df = FeatGen_CDV().transform(df, period=14, verbose=True, add_raw_delta=True)
    ########

    base_feat_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False, normalize_features=False)
    #base_feat_generator.fit_normalizing( mod_df )
    stub_generator = NoneGenerator(feat_list, n_points=256, flat_stack=False)

    opener_base_feat_generator = stub_generator
    buyer_feat_generator = stub_generator
    seller_base_feat_generator = stub_generator

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

    sample_generator = OnFlySupervisedSampleGenerator(symbol=symbol, timeframe=timeframe, base_generator=base_feat_generator)
    sample_generator.fit( train_df, composite_env,
                      norm_ema=0, smooth_ema=10,
                      start_hold_proba=0.9, end_hold_proba=0.95, n_estimates=100,
                      show_plots=False )

    save( sample_generator, os.path.join( "../../data/interim/sample_generator_{}_{}.pkl".format(symbol, timeframe) ) )

if __name__ == "__main__":
    start_time = datetime.now()
    print("start time: {}".format(start_time))

    symbols = ["USDJPYrfd", "AUDUSDrfd", "USDCADrfd", "NZDUSDrfd", "USDCHFrfd", "EURUSDrfd", "GBPUSDrfd"]
    timeframe = "M5"
    feat_list = ["open", "close", "low", "high", "tick_volume", "delta", "cdv"]

    Parallel(n_jobs=7)(delayed(make_sample_generator)( symbol, deepcopy(timeframe), deepcopy(feat_list) ) for symbol in symbols)

