from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, history_price_array, base_feature_generator, price_feature_names_dict, start_point):

        self.history_price_array = history_price_array
        self.base_feature_generator = base_feature_generator
        self.price_feature_names_dict = price_feature_names_dict
        self.start_point = start_point
        self.history_ids = []
        for i in range(start_point, len(history_price_array)):
            self.history_ids.append(i)
        self.history_ids = np.array(self.history_ids)

    def __len__(self):
        return len(self.history_ids)

    def __getitem__(self, idx):
        i = self.history_ids[idx]
        x_i = self.base_feature_generator.get_features(self.history_price_array, self.price_feature_names_dict, i)
        x_i = np.float32(x_i)

        # conv 1d
        #x_i = x_i.reshape((x_i.shape[1], x_i.shape[2]))

        # conv 2d
        x_i = x_i.reshape((1, x_i.shape[1], x_i.shape[2]))

        return x_i

class FeatGen_HistoryEmbedder():
    def __init__(self, base_feature_generator, embedder_model, device="cuda"):

        self.base_feature_generator = base_feature_generator
        self.embedder_model = embedder_model.to(device)
        self.device = device

        self.feature_list = base_feature_generator.feature_list
        self.feature_list_size = len(base_feature_generator.feature_list)
        self.n_points = base_feature_generator.n_points
        self.flat_stack = base_feature_generator.flat_stack
        self.feature_shape = None

        if isinstance( self.embedder_model, torch.nn.Sequential ):
            self.latent_size = self.embedder_model[-1].in_features
        else:
            self.latent_size = self.embedder_model.linear.in_features
        self.feature_shape = (1, self.latent_size)

        self.cached_embeddings = {}

        pass

    def fit(self, history_price_array, price_feature_names_dict, start_point, batch_size=16):

        history_obs_dataset = CustomDataset(history_price_array, self.base_feature_generator, price_feature_names_dict, start_point)
        history_ids = history_obs_dataset.history_ids
        history_obs_dataloader = DataLoader( history_obs_dataset, shuffle=False, batch_size=batch_size )
        history_embeddings = []
        for history_obs_batch in tqdm( history_obs_dataloader, desc="Making history embeddings" ):
            history_obs_batch = history_obs_batch.to( self.device )
            if isinstance(self.embedder_model, torch.nn.Sequential):
                embeddings_batch = self.embedder_model[:len(self.embedder_model)-1]( history_obs_batch )
            else:
                embeddings_batch = self.embedder_model.get_embeddings( history_obs_batch )
            embeddings_batch = embeddings_batch.detach().cpu().numpy()
            for j in range( len(embeddings_batch) ):
                history_embeddings.append( embeddings_batch[j] )
        history_embeddings = np.array( history_embeddings )

        for i in range(len(history_ids)):
            self.cached_embeddings[ history_ids[i] ] = history_embeddings[i]

        return self

    def get_feature_shape(self):
        return self.feature_shape

    def get_features(self, history_price_array, price_feature_names_dict, history_step_id, expand_dims=True):
        obs = self.cached_embeddings[ history_step_id ]

        return obs

    def get_feats_(self, history_price_array, price_feature_names_dict, history_step_id ):

        pass