
from classes.delta_bender.SymbolDataManager import SymbolDataManager
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV

from classes.delta_bender import PlotRender


symbol = "EURUSD_i"
timeframe = "M15"
dataManager = SymbolDataManager("../data/raw/")
df = dataManager.getData(symbol, timeframe)
print(df)

df = df.tail(1200)

df = FeatGen_HeikenAshi().transform(df, verbose=True)
#PlotRender().plot_candles(df)

df = FeatGen_CDV().transform(df, period=32, verbose=True)
#PlotRender().plot_cdv(df)

PlotRender().plot_price_cdv(df)

print("done")