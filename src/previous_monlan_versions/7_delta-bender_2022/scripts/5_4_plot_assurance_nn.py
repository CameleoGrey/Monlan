

from classes.delta_bender.ChangePredictor import ChangePredictor
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from classes.delta_bender.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from classes.delta_bender.FeatGen_PriceEMA import FeatGen_PriceEMA
from classes.delta_bender.PlotRender import PlotRender
from classes.delta_bender.SymbolDataManager import SymbolDataManager
from classes.multipurpose.utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, MSE, MeanAbsoluteError
from classes.delta_bender.ResnetBuilder import ResnetBuilder
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


symbol = "EURUSD_i"
timeframe = "M5"
target_columns = ["open", "high", "low", "close", "tick_volume"]
window_size = 64
ema_period_volume = 14
ema_period_price = 3

dataManager = SymbolDataManager("../data/raw/")
df = dataManager.getData(symbol, timeframe)
df = df[ target_columns ]
df = df.tail(1000)

df = FeatGen_HeikenAshi().transform(df)

df = FeatGen_CDV().transform(df, ema_period_volume)
df = df.iloc[ema_period_volume - 1:]

#df = FeatGen_PriceEMA().transform(df, ema_period_price, verbose=False)
#df = df.iloc[ema_period_price - 1:]

#df = FeatGen_CDV().transform(df, ema_period_volume)
#df = df.iloc[ema_period_volume - 1:]

del df["tick_volume"]

df_vals = df.values
print(df.shape)


model = ResnetBuilder().build_resnet_34( (1, 64, 5), num_outputs=2, outputActivation="sigmoid" )
model.compile( optimizer=Adam(learning_rate=0.0001), loss=CategoricalCrossentropy() )
model.load_weights( "../models/resnet_ha_no_cake_softmax" + ".h5" )
feat_gen = FeatGen_ScaledWindow(["open", "high", "low", "close", "cdv"], nPoints=window_size, nDiffs=0, flatStack=False)

pred_probas = []
stub_vals = [[0.0, 0.0] for i in range(window_size)]
for i in tqdm( range( df.index.values[0], df.index.values[-1] - window_size + 1), desc="generating dataset", colour="green" ):
    ind = i + window_size - 1
    x_i = feat_gen.get_window( ind, df )
    y_pred = model.predict( x_i )[0]
    y_pred = list( y_pred )
    pred_probas.append( y_pred )
pred_probas = stub_vals + pred_probas
pred_probas = np.array( pred_probas )
df["assurance_bull"] = pred_probas[:, 1]
df["assurance_bear"] = pred_probas[:, 0]

PlotRender().plot_assurance(df)
print("done")


