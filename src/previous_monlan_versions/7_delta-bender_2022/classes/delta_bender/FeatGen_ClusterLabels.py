
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS


class FeatGen_ClusterLabels():
    def __init__(self):

        self.scaler = MinMaxScaler()
        self.clusterizer = KMeans(n_clusters=8, init="k-means++", n_jobs=8)
        #self.clusterizer = AgglomerativeClustering(n_clusters=8)
        #self.clusterizer = OPTICS(n_jobs=8)

        pass

    def fit(self, df_, exclude_cols = ["open", "high", "low", "close", "tick_volume"], verbose=False):
        df = df_.copy()

        df = df.drop( exclude_cols, axis=1 )
        vals = df.values
        vals = self.scaler.fit_transform(vals)
        self.clusterizer.fit(vals, verbose)

        return self

    def transform(self, df_, exclude_cols = ["open", "high", "low", "close", "tick_volume"], verbose=False):
        df = df_.copy()

        df = df.drop( exclude_cols, axis=1 )
        vals = df.values
        vals = self.scaler.transform(vals)

        y_labels = self.clusterizer.predict(vals)
        df_["cluster_label"] = y_labels
        df_["cluster_label"] = df_["cluster_label"].astype( int )

        return df_

    def fit_transform(self, df_, exclude_cols = ["open", "high", "low", "close", "tick_volume"], verbose=False):
        df_ = df_.copy()
        self.fit( df_, exclude_cols, verbose )
        df_ = self.transform( df_, exclude_cols, verbose )
        return df_