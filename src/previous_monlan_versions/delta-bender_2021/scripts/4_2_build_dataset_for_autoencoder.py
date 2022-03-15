from classes.delta_bender.SymbolDataManager import SymbolDataManager
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.FeatGen_CandleRelations import FeatGen_CandleRelations
from classes.delta_bender.FeatGen_ClusterLabels import FeatGen_ClusterLabels
from classes.multipurpose.utils import *
from classes.delta_bender.MT5Terminal import MT5Terminal
from classes.delta_bender.SymbolDataUpdater import SymbolDataUpdater
from classes.delta_bender.FeatGenMeta import FeatGenMeta
import pandas as pd
from tqdm import tqdm

symbols = ["EURUSD_i", "AUDCAD_i", "AUDJPY_i", "AUDUSD_i",
           "CADJPY_i", "EURCAD_i", "EURGBP_i", "EURNZD_i",
           "GBPUSD_i", "USDCAD_i", "USDJPY_i", "USDCHF_i",
           "USDRUB_i"]
timeframes = ["D1", "H4", "H1", "M15", "M10", "M5"]
target_columns = ["open", "high", "low", "close", "tick_volume"]

"""terminal = MT5Terminal(login=123456, server="broker-server", password="password")
dataUpdater = SymbolDataUpdater("../data/raw/")
for symbol in symbols:
    for timeframe in timeframes:
        print(symbol + ": " + timeframe)
        dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")"""

window_size = 200
ema_period = 64
n_price_feats = 10
m_cdv_feats = 44
feat_gen = FeatGenMeta(use_heiken_ashi=True)
dataManager = SymbolDataManager("../data/raw/")

for timeframe in timeframes:
    for symbol in symbols:
        print(symbol + ": " + timeframe)
        df = dataManager.getData(symbol, timeframe)
        df = df[ target_columns ]

        df_vals = df.values
        x = []
        for i in tqdm( range( len(df_vals) - window_size - 1 ), desc="generating dataset", colour="green" ):
            x_i = df_vals[i : i + window_size ]
            x_i = feat_gen.transform(x_i, ema_period=ema_period, n_price_feats=n_price_feats, m_cdv_feats=m_cdv_feats)
            x.append(x_i)
        x = np.array( x )
        print(x.shape)
        save(x, "../data/autoencoder_datasets/{}_{}.pkl".format( symbol, timeframe ))
print("done")







