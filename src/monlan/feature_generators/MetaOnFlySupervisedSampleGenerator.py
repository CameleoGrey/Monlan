
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from joblib import Parallel, delayed

from src.monlan.modular_agents.StochasticCloser import StochasticCloser
from src.monlan.feature_generators.NoneGenerator import NoneGenerator


class MetaOnFlySupervisedSampleGenerator():
    def __init__(self, base_generators):

        self.base_generators = base_generators
        self.symbols = []
        self.timeframes = []
        for base_generator in base_generators:
            self.symbols.append( base_generator.symbol )
            self.timeframes.append( base_generator.timeframe )


        self.feature_list = base_generators[0].feature_list
        self.feature_list_size = len( base_generators[0].feature_list )
        self.n_points = base_generators[0].n_points
        self.flat_stack = base_generators[0].flat_stack
        self.feature_shape = base_generators[0].feature_shape

        self.generator_mapping = None
        self.meta_step_ids = None
        self.common_step_ids = None
        self.common_targets = None
        self.init_meta_generator_()

        pass

    def init_meta_generator_(self):

        self.generator_mapping = []
        self.common_step_ids = []
        self.common_targets = []

        for i in range( len(self.base_generators) ):
            gen_id_array = np.zeros( (len(self.base_generators[i].common_step_ids,)), dtype=np.int )
            gen_id_array = gen_id_array + i
            self.generator_mapping.append(gen_id_array)
            self.common_step_ids.append( self.base_generators[i].common_step_ids )
            self.common_targets.append(self.base_generators[i].common_targets)

        self.generator_mapping = np.hstack( self.generator_mapping )
        self.common_step_ids = np.hstack(self.common_step_ids)
        self.meta_step_ids = np.array([i for i in range(len(self.common_step_ids))])
        self.common_targets = np.vstack(self.common_targets)

        pass

    def get_sample(self, meta_step_id):

        generator_id = self.generator_mapping[ meta_step_id ]
        history_step_id = self.common_step_ids[ meta_step_id ]

        x, y = self.base_generators[generator_id].get_sample( history_step_id )

        return x, y

    def get_train_val_split(self, test_size=0.03, shuffle=True):

        x_train, x_val, y_train, y_val = train_test_split( self.meta_step_ids, self.common_targets,
                                                           random_state=45, shuffle=shuffle, test_size=test_size)
        return x_train, x_val, y_train, y_val

    def show_plots(self, reward_line_plot=True, reward_dist_plot=True):

        for base_generator in self.base_generators:
            base_generator.show_plots( reward_line_plot, reward_dist_plot )

        pass