

################################################################
# Run closer training on meta on-fly sample generator (big data)
# to use in the next scripts.
################################################################

from src.monlan.modular_agents.BigSupervisedTCNMirrorCloser import BigSupervisedTCNMirrorCloser
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
        base_sample_generator = load( os.path.join( "../../data/interim/sample_generator_{}_{}.pkl".format(symbol, timeframe) ) )
        base_sample_generator.base_generator.normalize_features = False
        base_sample_generator.remake_history_step_target_mapping()
        base_sample_generators.append( base_sample_generator )
    sample_generator = MetaOnFlySupervisedSampleGenerator(base_sample_generators)
    #sample_generator.show_plots( reward_line_plot=True, reward_dist_plot=True )

    big_supervised_closer = BigSupervisedTCNMirrorCloser(input_size=7)
    big_supervised_closer.fit(sample_generator, n_jobs=4, val_size=0.03, val_batch_size=256,
                              epochs = 52, warm_up_epochs = 2,
                              batch_size = 256, learning_rate = 0.0001, batch_norm_momentum=0.1,
                              checkpoint_save_path = os.path.join("../../models", "best_val_resnet_closer.pkl"),
                              continue_train = True)