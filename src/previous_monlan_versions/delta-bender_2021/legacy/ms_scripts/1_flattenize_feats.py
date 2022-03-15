
from monlan.utils.save_load import *
from monlan.agents.ResnetBuilder import ResnetBuilder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from monlan_supervised.FeatureFlattener import FeatureFlattener
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import gc
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
import seaborn as sns

def loadDataSet(dir, agentType, symbol, timeframe):
    x = load("{}{}_x_{}_{}.pkl".format(dir, agentType, symbol, timeframe))
    y = load("{}{}_y_{}_{}.pkl".format(dir, agentType, symbol, timeframe))
    gc.collect()
    return x, y

###############################################################################

targetSymbol = "AUDUSD_i"
samplesDir = "../data/train_samples/"
timeframe = "M10"
agentType = "opener"
colBatchSize = 4096
colEpochs = 2
colLr = 0.001
vecLatentDim = 64
vecBatchSize = 64
vecEpochs = 10
vecLr = 0.001

x, y = loadDataSet(samplesDir, agentType, targetSymbol, timeframe)

##############################
#x = x[int(0.99 * len(x)):]
#save("../data/flattener_dev_set.pkl", x)
#x = load("../data/flattener_dev_set.pkl")
##############################

#flattener = FeatureFlattener()
#colSet = flattener.makeColumnsVecSet(x)
#flattener.buildColEncoder(colDim=colSet.shape[1], lr=colLr)
#flattener.fitColEncoder(colSet, batchSize=colBatchSize, epochs=colEpochs)
#flattener.saveColEncoder("../models/", "flattener_{}_{}".format(targetSymbol, timeframe))

flattener = FeatureFlattener()
flattener.loadColEncoder("../models/", "flattener_{}_{}".format(targetSymbol, timeframe))
x = flattener.flattenizeFeats(x)
flattener.buildVecEncoder(inputVecDim=x.shape[1], latentDim=vecLatentDim, lr=vecLr)
flattener.fitVecEncoder(x, batchSize=vecBatchSize, epochs=vecEpochs)
flattener.saveVecEncoder("../models/", "flattener_{}_{}".format(targetSymbol, timeframe))

flattener.loadVecEncoder("../models/", "flattener")
x = flattener.compressVecs(x, batchSize=vecBatchSize, verbose=1)
save("../data/train_samples/{}_x_{}_{}_flattened.pkl".format(agentType, targetSymbol, timeframe), x)
print("done")