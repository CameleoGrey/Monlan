
from monlan.utils.save_load import *
from sklearn.preprocessing import MinMaxScaler
from monlan_supervised.BVAE import BVAE
from tqdm import tqdm

import gc
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
import seaborn as sns

def loadDataSet(dir, agentType, symbol, timeframe):
    x = load("{}2d_feat_images_{}_{}_{}.pkl".format(dir, agentType, symbol, timeframe))
    #x = x[int(0.5 * len(x)):]
    y = load("{}2d_feat_targets_{}_{}_{}.pkl".format(dir, agentType, symbol, timeframe))
    #y = y[int(0.5 * len(y)):]
    gc.collect()
    return x, y

addSymbols = ["AUDUSD_i"]
#addSymbols = ["AUDUSD_i", "EURAUD_i", "USDCAD_i", "EURCAD_i", "GBPUSD_i"]
targetSymbol = "AUDUSD_i"
samplesDir = "../data/train_samples/"
timeframe = "M10"
agentTypes = [
    "opener",
    #"buyer",
    #"seller",
    #"opener"
]

for agentType in agentTypes:

    x_train, y_train = loadDataSet(samplesDir, agentType, targetSymbol, timeframe)
    #x_train = x_train[380000:]
    #y_train = y_train[380000:]
    scaler = MinMaxScaler()
    for i in tqdm(range(len(x_train)), desc="Scaling"):
        tmp = scaler.fit_transform(x_train[i].reshape((x_train[i].shape[0], x_train[i].shape[1])))
        tmp = tmp.reshape((tmp.shape + (1,)))
        x_train[i] = tmp

    bvae = BVAE().buildModel(x_train[0].shape, 100)
    bvae.fit(x_train, batch_size=64, epochs=3)

    encodedFeats = bvae.encode(x_train, batchSize=64)
    save("../data/train_samples/darknet_vecs_{}_{}_{}.pkl".format(agentType, targetSymbol, timeframe), encodedFeats)
    save("../data/train_samples/darknet_targets_{}_{}_{}.pkl".format(agentType, targetSymbol, timeframe), y_train)

print("done")