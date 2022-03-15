
import pandas as pd
import numpy as np
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

class FeatGenMeta():
    def __init__(self, use_heiken_ashi):

        self.needful_cols = ["open", "high", "low", "close", "tick_volume"]
        if use_heiken_ashi:
            self.gen_hk = FeatGen_HeikenAshi()
        else:
            self.gen_hk = None
        self.gen_cdv = FeatGen_CDV()

        pass

    def transform(self, x, ema_period, n_price_feats=None, m_cdv_feats=None):

        if isinstance(x, np.ndarray):
            x = pd.DataFrame( x, columns=self.needful_cols )
        elif isinstance(x, pd.DataFrame):
            x.reset_index(drop=True, inplace=True)

        if self.gen_hk is not None:
            x = self.gen_hk.transform(x)

        x = self.gen_cdv.transform(x, period=ema_period, verbose=False)
        x = x[ema_period - 1: ]
        x.reset_index(drop=True, inplace=True)

        cdv_vals = x["cdv"].values
        cdv_change_ind = []
        for i in range( 1, len(cdv_vals) ):

            cdv_change_ind.append( cdv_vals[i] - cdv_vals[i - 1] )
            """if cdv_vals[i] > cdv_vals[i-1]:
                cdv_change_ind.append( 1 )
            elif cdv_vals[i] < cdv_vals[i-1]:
                cdv_change_ind.append( -1 )
            else:
                cdv_change_ind.append( 0 )"""
        cdv_change_ind = np.array( cdv_change_ind ).reshape((-1, 1))
        cdv_change_ind = MinMaxScaler(feature_range=(-1, 1)).fit_transform(cdv_change_ind)
        cdv_change_ind = np.array(cdv_change_ind).reshape((-1,))

        x = x[1:]
        x.reset_index(drop=True, inplace=True)
        x["cdv_change_ind"] = cdv_change_ind

        cdv_vals = np.array(cdv_vals).reshape((-1, 1))
        cdv_vals = MinMaxScaler(feature_range=(-1, 1)).fit_transform(cdv_vals)
        cdv_vals = np.array(cdv_vals).reshape((-1,))
        if m_cdv_feats is not None:
            cdv_vals = cdv_vals[-m_cdv_feats:]
            cdv_change_feats = x["cdv_change_ind"].values[-m_cdv_feats:]
        else:
            cdv_change_feats = x["cdv_change_ind"].values

        price_feats = self.get_price_feats_(x, n_price_feats)

        feats = np.hstack( [price_feats, cdv_vals, cdv_change_feats] )

        return feats

    def get_price_feats_(self, df, n):

        df = df.copy()

        if n is not None:
            df = df.tail(n)

        a = np.array( [i for i in range(n)] ).reshape((-1, 1))
        x = np.vstack([a, a, a, a])
        y = np.hstack([df["open"].values, df["high"].values, df["low"].values, df["close"].values])
        y = np.array(y).reshape((-1, 1))
        y = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y)
        y = np.array(y).reshape((-1,))

        model = Ridge(random_state=45).fit(x, y)
        y_pred = model.predict( a )

        scaled_prices = []
        for i in range(4):
            price_property = y[i*n:(i+1)*n]
            scaled_prices.append( price_property )
        scaled_prices = np.array( scaled_prices )

        price_deltas = scaled_prices - y_pred

        means = np.mean( price_deltas, axis=0 )
        stds = np.std( price_deltas, axis=0 )
        min_deltas = np.min(price_deltas, axis=0)
        max_deltas = np.max(price_deltas, axis=0)

        std_global = np.std( price_deltas )
        min_deltas_global = np.min(price_deltas)
        max_deltas_global = np.max(price_deltas)
        angle = self.get_angle_(0.0, y_pred[0], 1.0, y_pred[1])

        price_feats = np.hstack( [y, means, stds, min_deltas, max_deltas,
                                  [std_global, min_deltas_global, max_deltas_global, angle]] )

        return price_feats

    def get_angle_(self, x_1, y_1, x_2, y_2):

        x = x_2 - x_1
        y = y_2 - y_1

        cos_x = x / np.sqrt( np.square(x) + np.square(y) )

        sin_x = np.sqrt(1 - np.square(cos_x))

        return sin_x
