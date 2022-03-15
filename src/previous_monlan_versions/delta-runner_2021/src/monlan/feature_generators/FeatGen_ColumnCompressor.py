import os.path

import numpy as np
from tqdm import tqdm
from datetime import datetime

from src.monlan.feature_generators.DenseAutoEncoder import DenseAutoEncoder
from tensorflow.keras.models import model_from_json, load_model
from src.monlan.utils.save_load import *

class FeatGen_ColumnCompressor():
    def __init__(self, feat_gen_2d=None):
        self.src_feat_gen = feat_gen_2d
        self.column_encoder = DenseAutoEncoder()

        self.cached_vectors = {}
        self.active_composite_env = None

        pass

    def fit(self, history_df, batch_size, epochs, lr=0.001):

        input_dim = len(self.src_feat_gen.featureList)
        self.column_encoder = self.column_encoder.buildModel(inputDim=input_dim, latentDim=1, lr=lr)

        price_feature_names_dict = {}
        price_feature_names = list(history_df.columns)
        for i in range(len(price_feature_names)):
            price_feature_names_dict[ price_feature_names[i]] = i
        history_price_array = history_df.copy().values

        training_data = []
        feature_shape = self.src_feat_gen.get_feature_shape()
        window_width = feature_shape[1]
        start_point = window_width - 1
        for i in tqdm(range(start_point, len(history_df)), desc="Making training data for column compressor"):
            feature_window = self.src_feat_gen.get_features( history_price_array, price_feature_names_dict, i, expandDims=False)
            feature_window = feature_window.reshape( (feature_window.shape[0], feature_window.shape[1]) )
            feature_window = feature_window.T
            column_features_batch = feature_window.reshape( feature_window.shape + (1,) )
            training_data.append( column_features_batch )
        training_data = np.vstack(training_data)

        self.column_encoder.fit( training_data, batchSize=batch_size, epochs=epochs )

        feat_extr_json = self.column_encoder.encoder.to_json()
        encoder_weigths = self.column_encoder.encoder.get_weights()
        featExtr = model_from_json(feat_extr_json)
        featExtr.set_weights(encoder_weigths)
        featExtr.compile(optimizer="adam", loss="mse")
        self.column_encoder = featExtr

        return self

    def cache_vectors(self, history_price_array, price_feature_names_dict, env_name, batch_size=2048):
        all_cols_data = []
        feature_shape = self.src_feat_gen.get_feature_shape()
        window_width = feature_shape[1]
        start_point = window_width - 1
        for i in tqdm(range(start_point, len(history_price_array)), desc="Getting window features to compress and cache"):
            feature_window = self.src_feat_gen.get_features( history_price_array, price_feature_names_dict, i, expandDims=False)
            feature_window = feature_window.reshape( (feature_window.shape[0], feature_window.shape[1]) )
            feature_window = feature_window.T
            column_features_batch = feature_window.reshape( feature_window.shape + (1,) )
            all_cols_data.append( column_features_batch )
        all_cols_data = np.vstack(all_cols_data)

        compressed_cols_feats = self.column_encoder.predict( all_cols_data, batch_size=2048, verbose=1 )
        vectors_dict = {}
        for i, j in tqdm(zip(range(start_point, len(history_price_array)), range(len(history_price_array) - window_width + 1)), desc="Caching vectors"):
            feat_vector = compressed_cols_feats[ j*window_width : (j+1)*window_width ]
            feat_vector = feat_vector.T
            vectors_dict[i] = feat_vector
        self.cached_vectors[env_name] = vectors_dict
        pass


    def set_active_env(self, env_name):
        self.active_composite_env = env_name
        pass



    def get_features(self, history_price_array, price_feature_names_dict, history_step_id, expandDims=True):

        if self.active_composite_env is not None \
                and self.active_composite_env in self.cached_vectors.keys():
            cached_vector = self.cached_vectors[self.active_composite_env][history_step_id]
            return cached_vector

        feature_window = self.src_feat_gen.get_features( history_price_array, price_feature_names_dict, history_step_id, expandDims=False)
        feature_window = feature_window.reshape( (feature_window.shape[0], feature_window.shape[1]) )
        feature_window = feature_window.T
        column_features_batch = feature_window.reshape( feature_window.shape + (1,) )
        compressed_features = self.column_encoder.predict( column_features_batch, batch_size=2048 )
        compressed_features = compressed_features.T

        return compressed_features

    def get_feature_shape(self):

        base_shape = self.src_feat_gen.get_feature_shape()
        compressed_shape = (base_shape[1],)

        return compressed_shape

    def save(self, dir, base_name):

        feat_extr_json = self.column_encoder.to_json()
        encoder_weigths = self.column_encoder.get_weights()
        encoder_network = {"architecture": feat_extr_json,
                           "weights": encoder_weigths}
        save(encoder_network, os.path.join(dir, base_name + "_nn.pkl"))

        encoder = self.column_encoder
        self.column_encoder = None
        save(self, os.path.join(dir, base_name + "_body.pkl"))
        self.column_encoder = encoder

        pass

    def load(self, dir, base_name):

        feature_compressor = load( os.path.join(dir, base_name + "_body.pkl") )

        encoder_network = load( os.path.join(dir, base_name + "_nn.pkl") )
        encoder = model_from_json(encoder_network["architecture"])
        encoder.set_weights(encoder_network["weights"])
        encoder.compile(optimizer="adam", loss="mse")

        feature_compressor.column_encoder = encoder

        return feature_compressor