
################################################################
# Run closer training on one of on-fly sample generators to test.
################################################################

from src.monlan.modular_agents.BigSupervisedTCNMirrorCloser import BigSupervisedTCNMirrorCloser

from datetime import datetime
import os
import numpy as np
from src.monlan.utils.save_load import *
from tqdm import tqdm

if __name__ == "__main__":
    sample_generator = load( os.path.join( "../../data/interim/sample_generator.pkl" ) )
    sample_generator.base_generator.normalize_features = False
    sample_generator.remake_history_step_target_mapping()
    big_supervised_closer = BigSupervisedTCNMirrorCloser(input_size=7)
    big_supervised_closer.fit(sample_generator, n_jobs=4,
                              epochs = 52, warm_up_epochs = 2, batch_size = 256, learning_rate = 0.001, batch_norm_momentum=0.1,
                             checkpoint_save_path = os.path.join("../../models", "best_val_supervised_mirror_closer.pkl"),
                              continue_train=False)
    save( big_supervised_closer, os.path.join("../../models", "supervised_mirror_closer.pkl") )