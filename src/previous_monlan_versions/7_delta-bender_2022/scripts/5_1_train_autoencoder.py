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
from classes.multipurpose.DenseAutoEncoder import DenseAutoEncoder
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#symbols = ["EURUSD_i", "AUDCAD_i", "AUDJPY_i", "AUDUSD_i",
#           "CADJPY_i", "EURCAD_i", "EURGBP_i", "EURNZD_i",
#           "GBPUSD_i", "USDCAD_i", "USDJPY_i", "USDCHF_i"]
symbols = ["EURUSD_i"]
timeframes = ["D1", "H4", "H1", "M15", "M10", "M5"]
#timeframes = ["H1"]

x_autoenc = []
for timeframe in timeframes:
    for symbol in symbols:
        x = load("../data/autoencoder_datasets/{}_{}.pkl".format( symbol, timeframe ))
        x_autoenc.append( x )
x_autoenc = np.vstack( x_autoenc )
x_autoenc = x_autoenc.reshape(x_autoenc.shape + (1,))

autoencoder = DenseAutoEncoder()
autoencoder.buildModel(inputDim=len(x_autoenc[0]), latentDim=100, lr=0.001)
print(x_autoenc.shape)
autoencoder.fit(x_autoenc, batchSize=128, epochs=60)
autoencoder.save_encoder("../models/", "encoder")
print("done")







