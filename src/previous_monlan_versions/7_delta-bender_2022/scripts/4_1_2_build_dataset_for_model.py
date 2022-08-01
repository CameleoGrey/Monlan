from classes.delta_bender.SymbolDataManager import SymbolDataManager
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.FeatGen_CandleRelations import FeatGen_CandleRelations
from classes.delta_bender.FeatGen_ClusterLabels import FeatGen_ClusterLabels
from classes.delta_bender.FeatGenMeta import FeatGenMeta
from classes.multipurpose.utils import *
from classes.delta_bender.PlotRender import PlotRender
import pandas as pd
from tqdm import tqdm

symbol = "EURUSD_i"
timeframe = "M5"
target_columns = ["open", "high", "low", "close", "tick_volume"]
dataManager = SymbolDataManager("../data/raw/")
df = dataManager.getData(symbol, timeframe)
df = df[ target_columns ]
#df = df.tail(10000)
print(df.shape)

#gen_hk = FeatGen_HeikenAshi()
gen_cr = FeatGen_CandleRelations()
gen_cluster = FeatGen_ClusterLabels()
#df = gen_hk.transform( df )
df_for_clusterizer = gen_cr.fit_transform( df )
gen_cluster.fit( df_for_clusterizer, verbose=True )
save( gen_cr, "../models/price_relator.pkl" )
save( gen_cluster, "../models/clusteriser.pkl" )

window_size = 200
ema_period = 64
n_price_feats = 10
m_cdv_feats = 44
feat_gen = FeatGenMeta(use_heiken_ashi=True)
change_percentile = 95

df_vals = df.values

change_deltas = []
for i in range( 1, len(df_vals) ):
    prev_candle = df_vals[i - 1]
    current_candle = df_vals[i]
    current_close_delta = abs( current_candle[3] - prev_candle[3] )
    change_deltas.append( current_close_delta )
#change_treshold = np.percentile(change_deltas, change_percentile)
change_deltas = pd.Series(change_deltas)
quantiles = pd.qcut( change_deltas, 20, retbins=True )[1]


x = []
y = []
for i in tqdm( range( len(df_vals) - window_size - 2 ), desc="generating dataset", colour="green" ):

    prev_candle = df_vals[ i + window_size ]
    current_candle = df_vals[ i + window_size + 1 ]
    current_delta = current_candle[3] - prev_candle[3]
    current_delta_sign = np.sign( current_delta )
    abs_delta = abs( current_delta )
    #current_delta = abs(((current_candle[3] - prev_candle[3]) + (current_candle[0] - prev_candle[0])) / 2.0)

    if abs_delta < quantiles[-2]:
        y_i = 0
    elif current_delta_sign == -1:
        y_i = -1
    else:
        y_i = 1

    if y_i in [-1, 1]:

        #########
        #tmp = df_vals[i + 1: i + window_size + 2]
        #tmp = pd.DataFrame( tmp, columns = target_columns )
        #tmp = FeatGen_CDV().transform(tmp, period=64)
        #PlotRender().plot_price_cdv( tmp )
        #########

        y.append(y_i)
        x_i = df_vals[i + 1: i + window_size + 1]
        x_i = feat_gen.transform(x_i, ema_period=ema_period, n_price_feats=n_price_feats, m_cdv_feats=m_cdv_feats)
        x.append(x_i)

x = np.array( x )
y = np.array( y )

print(x.shape)

save( x, "../data/x_for_model.pkl" )
save( y, "../data/y_for_model.pkl" )
print("done")






