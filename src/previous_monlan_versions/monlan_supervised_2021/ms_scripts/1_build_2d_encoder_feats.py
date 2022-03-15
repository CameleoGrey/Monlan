
from monlan.utils.save_load import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

def loadDataSet(dir, agentType, symbol, timeframe):
    x = load("{}{}_x_{}_{}_flattened.pkl".format(dir, agentType, symbol, timeframe))
    y = load("{}{}_y_{}_{}.pkl".format(dir, agentType, symbol, timeframe))
    gc.collect()
    return x, y

addSymbols = ["AUDUSD_i"]
targetSymbol = "AUDUSD_i"
samplesDir = "../data/train_samples/"
timeframe = "M10"
agentTypes = [
    "opener",
]


for agentType in agentTypes:
    x_train, y_train = loadDataSet(samplesDir, agentType, targetSymbol, timeframe)
    nRows = x_train[0].shape[0]

    targets2D = y_train[nRows:]
    save("../data/train_samples/2d_feat_targets_{}_{}_{}.pkl".format(agentType, targetSymbol, timeframe), targets2D)

    imgFeats = np.zeros((x_train.shape[0] - nRows, nRows, nRows, 1), dtype=np.float32)
    print(targets2D.shape)
    print(imgFeats.shape)

    for i in tqdm(range(nRows, len(x_train))):
        featImg = x_train[i-nRows: i]
        featImg = featImg.reshape( (featImg.shape + (1,)) )
        imgFeats[i-nRows] = featImg
    tmp = imgFeats[-1]
    del x_train
    gc.collect()

    save("../data/train_samples/2d_feat_images_{}_{}_{}.pkl".format(agentType, targetSymbol, timeframe), imgFeats)
    del imgFeats
    gc.collect()

print("done")