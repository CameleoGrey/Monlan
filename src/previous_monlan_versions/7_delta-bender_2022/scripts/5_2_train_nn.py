from classes.delta_bender.SymbolDataManager import SymbolDataManager
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.FeatGen_CandleRelations import FeatGen_CandleRelations
from classes.delta_bender.FeatGen_ClusterLabels import FeatGen_ClusterLabels
from classes.multipurpose.utils import *
from sklearn.model_selection import train_test_split
from classes.delta_bender.MT5Terminal import MT5Terminal
from classes.delta_bender.SymbolDataUpdater import SymbolDataUpdater
from classes.delta_bender.FeatGenMeta import FeatGenMeta
import pandas as pd
from tqdm import tqdm
from classes.multipurpose.DenseAutoEncoder import DenseAutoEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from classes.delta_bender.ResnetBuilder import ResnetBuilder
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def preproc_y( y ):
    y_preproc = []
    for i in range(len(y)):
        if y[i] == -1:
            y_preproc.append([1.0, 0.0])
        else:
            y_preproc.append([0.0, 1.0])
    y = np.array(y_preproc)
    return y

#symbols = ["EURUSD_i"]
symbols = ["EURUSD_i", "AUDCAD_i", "AUDJPY_i", "AUDUSD_i",
           "CADJPY_i", "EURCAD_i", "EURGBP_i", "EURNZD_i",
           "GBPUSD_i", "USDCAD_i", "USDJPY_i", "USDCHF_i"]

#timeframes = ["D1", "H4", "H1", "M15", "M10", "M5"]
timeframes = ["M5"]
#timeframes = ["D1"]

x_train = []
y_train = []
x_test = []
y_test = []
for timeframe in timeframes:
    for symbol in symbols:
        x_loaded = load("../data/model_datasets/x_{}_{}.pkl".format( symbol, timeframe ))
        y_loaded = load("../data/model_datasets/y_{}_{}.pkl".format(symbol, timeframe))

        n_train = int(0.9*len( x_loaded ))
        x_loaded_train = x_loaded[ : n_train]
        x_loaded_test = x_loaded[n_train : ]
        y_loaded_train = y_loaded[: n_train]
        y_loaded_test = y_loaded[n_train : ]

        x_train.append( x_loaded_train )
        x_test.append( x_loaded_test )
        y_train.append( y_loaded_train )
        y_test.append( y_loaded_test )

x_train = np.vstack( x_train )
y_train = np.hstack( y_train )
x_test = np.vstack( x_test )
y_test = np.hstack( y_test )

x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, stratify=y_train, test_size=0.05, shuffle=True, random_state=45 )
y_train = preproc_y( y_train )
y_val = preproc_y( y_val )
y_test = preproc_y( y_test )

print( x_train.shape )
print( x_val.shape )
print( x_test.shape )

my_callbacks = [
        EarlyStopping(patience=5),
        ModelCheckpoint(filepath="../models/resnet_ha_cake_softmax" + ".h5",
                        save_weights_only=True,
                        save_best_only=True,
                        monitor="val_loss",
                        mode="min",
                        verbose=1)
]

model = ResnetBuilder().build_resnet_34( (1, x_train[0].shape[0], x_train[0].shape[1]), num_outputs=2, outputActivation="softmax" )
model.compile( optimizer=Adam(learning_rate=0.001), loss=CategoricalCrossentropy() )
model.fit( x_train, y_train, validation_data=(x_val, y_val),
           batch_size=128, epochs=100, shuffle=True, callbacks=my_callbacks)


#model = ResnetBuilder().build_resnet_34( (1, x_train[0].shape[0], x_train[0].shape[1]), num_outputs=2, outputActivation="softmax" )
#model.compile( optimizer=Adam(learning_rate=0.0001), loss=CategoricalCrossentropy() )
#model.load_weights( "../models/resnet_ha_cake_softmax" + ".h5" )

y_pred = model.predict( x_test, batch_size=128, verbose=1 )

y_pred_discretized = []
for i in range( len(y_pred) ):
    if y_pred[i][0] > y_pred[i][1]:
        y_i = -1
    else:
        y_i = 1
    y_pred_discretized.append( y_i )
y_pred_discretized = np.array( y_pred_discretized )


y_test_discretized = []
for i in range( len(y_test) ):
    if y_test[i][0] > y_test[i][1]:
        y_i = -1
    else:
        y_i = 1
    y_test_discretized.append( y_i )
y_test_discretized = np.array( y_test_discretized )

scores = get_all_scores( y_pred_discretized, y_test_discretized )
print( scores )


print("done")







