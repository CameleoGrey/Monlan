
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


class FeatGen_CandleRelations():
    def __init__(self):

        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.relation_scaler = MinMaxScaler( feature_range=(-1, 1) )

        pass

    def fit_transform(self, df):
        df = df.copy()
        self.fit( df )
        df = self.transform( df, verbose=False )
        return df

    def fit(self, df):
        df = df.copy()

        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        open = np.reshape(open, newshape=(-1, 1))
        high = np.reshape(high, newshape=(-1, 1))
        low = np.reshape(low, newshape=(-1, 1))
        close = np.reshape(close, newshape=(-1, 1))

        self.price_scaler.partial_fit(open)
        self.price_scaler.partial_fit(high)
        self.price_scaler.partial_fit(low)
        self.price_scaler.partial_fit(close)

        open = self.price_scaler.transform(open)
        high = self.price_scaler.transform(high)
        low = self.price_scaler.transform(low)
        close = self.price_scaler.transform(close)

        rel_df = self.get_relations(open, high, low, close)
        self.relation_scaler.fit( rel_df.values )

        return self

    def transform(self, df, verbose=False):
        df = df.copy()
        df.reset_index(drop=True, inplace=True)

        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        open = np.reshape( open, newshape=(-1, 1) )
        high = np.reshape( high, newshape=(-1, 1) )
        low = np.reshape( low, newshape=(-1, 1) )
        close = np.reshape( close, newshape=(-1, 1) )

        open = self.price_scaler.transform( open )
        high = self.price_scaler.transform( high )
        low = self.price_scaler.transform( low )
        close = self.price_scaler.transform( close )

        open = np.reshape(open, newshape=(-1,))
        high = np.reshape(high, newshape=(-1,))
        low = np.reshape(low, newshape=(-1,))
        close = np.reshape(close, newshape=(-1,))

        rel_df = self.get_relations(open, high, low, close)
        scaled_rels = self.relation_scaler.transform( rel_df.values )
        for col, i in zip( rel_df.columns, range(scaled_rels.shape[1]) ):
            rel_df[col] = scaled_rels[:, i]

        df = pd.concat( [df, rel_df], axis=1 )

        return df

    def get_relations(self, open, high, low, close ):
        open = np.reshape(open, newshape=(-1,))
        high = np.reshape(high, newshape=(-1,))
        low = np.reshape(low, newshape=(-1,))
        close = np.reshape(close, newshape=(-1,))

        full_candle_len = []
        body_len = []

        low_body_rel = []
        low_body_abs_rel = []
        low_full_candle_rel = []
        low_high_abs_rel = []
        low_high_rel = []

        high_body_rel = []
        high_body_abs_rel = []
        high_full_candle_rel = []
        high_low_abs_rel = []
        high_low_rel = []

        sum_high_low_full_candle_abs_rel = []
        sum_high_low_full_candle_rel = []
        sum_high_low_body_abs_rel = []
        sum_high_low_body_rel = []

        for i in range(len(open)):
            full_candle_len.append(abs(high[i] - low[i]))
            body_len.append(abs(close[i] - open[i]))

            high_body_rel.append(high[i] / body_len[i])
            high_body_abs_rel.append(abs(high[i] / body_len[i]))
            high_full_candle_rel.append(high[i] / full_candle_len[i])
            high_low_abs_rel.append(abs(high[i] / low[i]))
            high_low_rel.append(high[i] / low[i])

            low_body_rel.append(low[i] / body_len[i])
            low_body_abs_rel.append(abs(low[i] / body_len[i]))
            low_full_candle_rel.append(low[i] / full_candle_len[i])
            low_high_abs_rel.append(abs(low[i] / high[i]))
            low_high_rel.append(low[i] / high[i])

            sum_high_low_full_candle_abs_rel.append(abs((high[i] + low[i]) / full_candle_len[i]))
            sum_high_low_full_candle_rel.append((high[i] + low[i]) / full_candle_len[i])
            sum_high_low_body_abs_rel.append(abs((high[i] + low[i]) / body_len[i]))
            sum_high_low_body_rel.append((high[i] + low[i]) / body_len[i])


        df = pd.DataFrame()

        df["fcl"] = full_candle_len
        df["bl"] = body_len

        df["hbr"] = high_body_rel
        df["hbar"] = high_body_abs_rel
        df["hfcr"] = high_full_candle_rel
        df["hlar"] = high_low_abs_rel
        df["hlr"] = high_low_rel

        df["lbr"] = low_body_rel
        df["lbar"] = low_body_abs_rel
        df["lfcr"] = low_full_candle_rel
        df["lhar"] = low_high_abs_rel
        df["lhr"] = low_high_rel

        df["shlfcar"] = sum_high_low_full_candle_abs_rel
        df["shlfcr"] = sum_high_low_full_candle_rel
        df["shlbar"] = sum_high_low_body_abs_rel
        df["shlbr"] = sum_high_low_body_rel

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.select_dtypes(include=np.number):
            df[col] = df[col].fillna(df[col].median())

        return df