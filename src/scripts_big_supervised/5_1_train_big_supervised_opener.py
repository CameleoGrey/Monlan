
################################################################
# Train a big supervised opener using on-fly sample generators
# from the previous steps.
################################################################

from src.monlan.modular_agents.BigSupervisedTCNMirrorOpener import BigSupervisedTCNMirrorOpener
from src.monlan.feature_generators.MetaOnFlySupervisedSampleGenerator import MetaOnFlySupervisedSampleGenerator

from datetime import datetime
import os
import numpy as np
from src.monlan.utils.save_load import *
from tqdm import tqdm

if __name__ == "__main__":

    symbols = ["USDJPYrfd", "AUDUSDrfd", "USDCADrfd", "NZDUSDrfd", "USDCHFrfd", "EURUSDrfd", "GBPUSDrfd"]
    timeframe = "M5"
    base_sample_generators = []
    for symbol in symbols:
        base_sample_generator = load( os.path.join( "../../data/interim/opener_sample_generator_{}_{}.pkl".format(symbol, timeframe) ) )
        base_sample_generator.base_generator.normalize_features = False
        base_sample_generator.remake_history_step_target_mapping()
        base_sample_generators.append( base_sample_generator )
    sample_generator = MetaOnFlySupervisedSampleGenerator(base_sample_generators)
    #sample_generator.show_plots( reward_line_plot=True, reward_dist_plot=True )

    big_supervised_closer = BigSupervisedTCNMirrorOpener(input_size=7)
    # Current realization's using pytorch DataLoader which creates new process for parallelization of
    # a task. It copies objects for each process.
    # For the current task copying consumes a lot of RAM and slows the start of
    # each training epoch. Try n_jobs=0 or n_jobs=1, then find optimal for your hardware.
    # For me (CameleoGrey) n_jobs=4 consumes around 55/64 Gb RAM and achieves maximum speed
    # of the on-fly sample generation.
    big_supervised_closer.fit(sample_generator, n_jobs=4, val_size=0.03, val_batch_size=256,
                              epochs = 52, warm_up_epochs = 2,
                              batch_size = 256, learning_rate = 0.0001, batch_norm_momentum=0.1,
                              checkpoint_save_path = os.path.join("../../models", "best_val_resnet_opener.pkl"),
                              continue_train = False)