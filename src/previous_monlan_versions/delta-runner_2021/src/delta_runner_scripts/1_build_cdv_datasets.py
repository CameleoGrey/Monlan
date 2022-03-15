
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.utils.save_load import *

startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = dataManager.getData(symbol, timeframe)
df = df.tail(400000)

########
mod_df = FeatGen_CDV().transform(df, period=32, verbose=True)

mod_df["cdv"] = mod_df["cdv"].apply(lambda x: np.sign(x) * np.log1p(np.abs(x)))

shifted = mod_df["cdv"].shift(1)
not_shifted = mod_df["cdv"]
mod_df["cdv"] = not_shifted - shifted
mod_df.dropna(inplace=True)

#mod_df["cdv"].plot()
#plt.show()

#plot_render = PlotRender()
#plot_render.plot_price_cdv(mod_df)
#plot_render.plot_cdv(mod_df)
########

scaler = MinMaxScaler( feature_range=(-1, 1) )
scaler.fit( mod_df["cdv"].values.reshape((-1, 1)) )
save( scaler, os.path.join( "../../data", "scaler.pkl" ) )
mod_df["cdv"] = scaler.transform( mod_df["cdv"].values.reshape((-1, 1)) )

x = []
y = []
n_points = 128
cdv_points = mod_df["cdv"].values
for i in tqdm(range(n_points, len(cdv_points))):
    features_sample = cdv_points[ i - n_points : i ]
    target = cdv_points[i]
    x.append( features_sample )
    y.append( target )
x = np.array( x )
y = np.array( y )

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, shuffle=False )
x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=0.2, shuffle=True, random_state=45 )

dataset = {}
dataset["train"] = ( x_train, y_train )
dataset["test"] = ( x_test, y_test )
dataset["val"] = ( x_val, y_val )
save(dataset, os.path.join( "../../data", "dataset.pkl" ))

print( "done" )