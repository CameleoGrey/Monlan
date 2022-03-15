from classes.delta_bender.SymbolDataManager import SymbolDataManager
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.FeatGen_PriceEMA import FeatGen_PriceEMA
from classes.delta_bender.FeatGen_CandleRelations import FeatGen_CandleRelations
from classes.delta_bender.FeatGen_ClusterLabels import FeatGen_ClusterLabels
from classes.multipurpose.utils import *
from classes.delta_bender.MT5Terminal import MT5Terminal
from classes.delta_bender.SymbolDataUpdater import SymbolDataUpdater
from classes.delta_bender.FeatGen_ScaledWindow import FeatGen_ScaledWindow
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np

symbols = ["EURUSD_i", "AUDCAD_i", "AUDJPY_i", "AUDUSD_i",
           "CADJPY_i", "EURCAD_i", "EURGBP_i", "EURNZD_i",
           "GBPUSD_i", "USDCAD_i", "USDJPY_i", "USDCHF_i",
           "USDRUB_i"]
timeframes = ["D1", "H4", "H1", "M15", "M10", "M5"]
#timeframes = ["M5"]
target_columns = ["open", "high", "low", "close", "tick_volume"]

"""terminal = MT5Terminal(login=123456, server="broker-server", password="password")
dataUpdater = SymbolDataUpdater("../data/raw/")
for symbol in symbols:
    for timeframe in timeframes:
        print(symbol + ": " + timeframe)
        dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")"""

def proc_symbol_timeframe_pair( symbol, timeframe ):
    feat_window_size = 64
    ema_period_volume = 14
    ema_period_price = 3
    close_delta_window = 6
    feat_gen = FeatGen_ScaledWindow(["open", "high", "low", "close", "cdv"], nPoints=feat_window_size, nDiffs=0, flatStack=False)
    dataManager = SymbolDataManager("../data/raw/")

    print(symbol + ": " + timeframe)
    df = dataManager.getData(symbol, timeframe)
    df = df[target_columns]

    df = FeatGen_HeikenAshi().transform(df)

    df = FeatGen_CDV().transform(df, ema_period_volume)
    df = df.iloc[ema_period_volume - 1:]

    #df = FeatGen_PriceEMA().transform(df, ema_period_price, verbose=False)
    #df = df.iloc[ema_period_price - 1:]

    #df = FeatGen_CDV().transform(df, ema_period_volume)
    #df = df.iloc[ema_period_volume - 1:]

    del df["tick_volume"]

    ###############################
    df_vals = df.values
    change_deltas = []
    for i in range(1, len(df_vals) - close_delta_window):
        prev_candle = df_vals[i - 1]
        mean_delta = []
        for j in range(close_delta_window):
            current_candle = df_vals[i + j]
            mean_delta.append( abs(current_candle[3] - prev_candle[3]) )
        mean_delta = np.average( mean_delta )
        change_deltas.append(mean_delta)
    # change_treshold = np.percentile(change_deltas, change_percentile)
    change_deltas = pd.Series(change_deltas)
    quantiles = pd.qcut(change_deltas, 20, retbins=True)[1]
    ################################

    x = []
    y = []
    for i in tqdm(range(df.index.values[0] + 1, df.index.values[-1] - feat_window_size - close_delta_window - 1), desc="generating dataset",
                  colour="green"):
        current_candle = df.loc[i + feat_window_size]
        mean_delta = []
        for j in range(close_delta_window):
            next_candle = df.loc[i + j + feat_window_size + 1]
            mean_delta.append(next_candle[3] - current_candle[3])
        mean_delta = np.average(mean_delta)
        current_delta_sign = np.sign(mean_delta)
        abs_delta = abs(mean_delta)
        # current_delta = abs(((current_candle[3] - prev_candle[3]) + (current_candle[0] - prev_candle[0])) / 2.0)

        min_delta_treshold = quantiles[-7] #70%
        max_delta_treshold = quantiles[-2] #95%
        if abs_delta < min_delta_treshold:
            y_i = 0
        elif abs_delta > min_delta_treshold and abs_delta < max_delta_treshold and current_delta_sign == -1:
            y_i = -1
        elif abs_delta > min_delta_treshold and abs_delta < max_delta_treshold and current_delta_sign == 1:
            y_i = 1
        else:
            y_i = None

        if y_i in [-1, 1]:
            y.append(y_i)
            ind = i + feat_window_size
            ######
            # DEBUG
            #ind += 1
            ######
            x_i = feat_gen.get_window(ind, df)
            x.append(x_i)

    x = np.vstack(x)
    y = np.array(y)
    print(x.shape)
    save(x, "../data/model_datasets/x_{}_{}.pkl".format(symbol, timeframe))
    save(y, "../data/model_datasets/y_{}_{}.pkl".format(symbol, timeframe))


st_pairs = []
for i in range(len(symbols)):
    for j in range( len(timeframes) ):
        st_pairs.append( [symbols[i], timeframes[j]] )
st_pairs = np.array(st_pairs)
np.random.seed(45)
np.random.shuffle( st_pairs )

Parallel(n_jobs=13, verbose=100)(delayed(proc_symbol_timeframe_pair)(st_pair[0], st_pair[1]) for st_pair in st_pairs )


print("done")







