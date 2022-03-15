
from classes.delta_bender.SymbolDataManager import SymbolDataManager
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.FeatGen_CandleRelations import FeatGen_CandleRelations
from classes.delta_bender.FeatGen_ClusterLabels import FeatGen_ClusterLabels
from classes.delta_bender.PlotRender import PlotRender
import numpy as np

symbol = "EURUSD_i"
timeframe = "M15"
dataManager = SymbolDataManager("../data/raw/")
df = dataManager.getData(symbol, timeframe)
df = df[["open", "high", "low", "close", "tick_volume"]]

gen_hk = FeatGen_HeikenAshi()
gen_cr = FeatGen_CandleRelations()
gen_cluster = FeatGen_ClusterLabels()
#df = df.tail(5000)
df = gen_hk.transform( df )
df_for_clusterizer = gen_cr.fit_transform( df )
gen_cluster.fit( df_for_clusterizer, verbose=True )

#df = df.tail(1000)
df = gen_cr.transform(df)
df = gen_cluster.transform(df)
df = FeatGen_CDV().transform(df, period=32)

"""y_cluster = df["cluster_label"].valuesdf = df.drop( ["open", "high", "low", "close", "tick_volume", "cluster_label", "cdv"], axis=1 )
vals = df.values
uniq_y = np.unique( y_cluster )
x_tsne = TSNE(perplexity=60, n_iter=1000, n_jobs=8, verbose=True).fit_transform(vals)
for uy in uniq_y:
    plt.scatter( x_tsne[ y_cluster == uy, 0], x_tsne[ y_cluster == uy, 1], s=2 )
plt.show()"""

y_cluster = df["cluster_label"].values
uniq_y = np.unique( y_cluster )

print(len(uniq_y))
for uy in uniq_y:
    cluster_df = df[ df["cluster_label"] == uy ]
    cluster_df = cluster_df.iloc[:2000]
    PlotRender().plot_price_cdv( cluster_df )

print("done")