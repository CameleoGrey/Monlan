
from monlan.utils.save_load import *
from monlan.agents.ResnetBuilder import ResnetBuilder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import gc
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

def loadDataSet(dir, agentType, symbol, timeframe):
    #x = load("{}{}_x_{}_{}_flattened.pkl".format(dir, agentType, symbol, timeframe))
    # y = load("{}{}_y_{}_{}.pkl".format(dir, agentType, symbol, timeframe))
    x = load("../data/train_samples/darknet_vecs_{}_{}_{}.pkl".format(agentType, symbol, timeframe))
    y = load("../data/train_samples/darknet_targets_{}_{}_{}.pkl".format(agentType, symbol, timeframe))
    gc.collect()
    return x, y

def removeOutliers(x, y):
    stdY = np.std(y)
    meanY = np.mean(y)
    if agentType == "opener":
        print(y.shape)
        cleanedX = x[np.abs(y[:, 0]) < (meanY + 2 * stdY)]
        cleanedY = y[np.abs(y[:, 0]) < (meanY + 2 * stdY)]
        del x, y
        gc.collect()

        print(cleanedY.shape)
        cleanedX = cleanedX[np.abs(cleanedY[:, 2]) < (meanY + 2 * stdY)]
        cleanedY = cleanedY[np.abs(cleanedY[:, 2]) < (meanY + 2 * stdY)]
        gc.collect()

        print(cleanedY.shape)
        cleanedX = cleanedX[np.abs(cleanedY[:, 1]) < (meanY + 2 * stdY)]
        cleanedY = cleanedY[np.abs(cleanedY[:, 1]) < (meanY + 2 * stdY)]
        gc.collect()

        print(cleanedY.shape)
        print("cleaned outliers")
    else:
        #q1 = np.quantile(0.25)
        #q3 = np.quantile(0.75)
        #iqr = q3 - q1
        #lowerTail = q1 - 1.5 * iqr
        #upperTail = q3 + 1.5 * iqr
        print(y.shape)
        cleanedX = x[np.abs(y[:, 0]) < (meanY + 3 * stdY)]
        cleanedY = y[np.abs(y[:, 0]) < (meanY + 3 * stdY)]
        del x, y
        gc.collect()
        print(cleanedY.shape)

        cleanedX = cleanedX[np.abs(cleanedY[:, 1]) < (meanY + 3 * stdY)]
        cleanedY = cleanedY[np.abs(cleanedY[:, 1]) < (meanY + 3 * stdY)]
        print(cleanedY.shape)
    y = cleanedY
    x = cleanedX
    del cleanedX, cleanedY
    gc.collect()
    print(np.mean(y, axis=0))
    print(np.min(y, axis=0))
    print(np.max(y, axis=0))
    print(np.std(y, axis=0))
    return x, y

def scaleTargets(y):
    #scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    if agentType == "opener":
        scalerFitData = np.hstack([y[:, 0], y[:, 1], y[:, 2]])
    else:
        scalerFitData = np.hstack([y[:, 0], y[:, 1]])
    scalerFitData = scalerFitData.reshape(-1, 1)
    scaler.fit(scalerFitData)
    del scalerFitData
    gc.collect()
    y[:, 0] = scaler.transform(y[:, 0].reshape(-1, 1)).reshape(-1, )
    y[:, 1] = scaler.transform(y[:, 1].reshape(-1, 1)).reshape(-1, )
    if agentType == "opener":
        y[:, 2] = scaler.transform(y[:, 2].reshape(-1, 1)).reshape(-1, )
    return y

def logTargets(y):
    y[:, 0] = np.log1p(y[:, 0])
    y[:, 1] = np.log1p(y[:, 1])
    if agentType == "opener":
        y[:, 2] = np.log1p(y[:, 2])
    return y

def plotTargets(y):
    sampleY = np.random.randint(0, len(y), size=10000)
    sampleY = y[sampleY]
    for i in range(sampleY[0].shape[0]):
        #sns.kdeplot( x=[x for x in range(len(sampleY))], y=sampleY[:, i])
        sns.displot(x=sampleY[:, i])
    plt.show()
    pass

def adjustDataSet(x, y, agentType):
    if agentType == "opener":
        #y[:, 1] = 0.0 * (0.5 * (y[:, 0] + y[:, 2]))
        y[:, 1] = 4.0 * y[:, 1] #managing hold weight
        buyDiff = y[:, 0] - y[:, 2]
        sellDiff = y[:, 2] - y[:, 0]
        y[:, 0] = buyDiff
        y[:, 2] = sellDiff
        del buyDiff, sellDiff
        gc.collect()
    else:
        y[:, 1] = 4.0 * y[:, 1]
        pass

    #plotTargets(y)
    x, y = removeOutliers(x, y)
    #plotTargets(y)

    """for i in tqdm(range(len(y))):
        tmp = y[i].reshape((-1, 1))
        tmp = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(tmp)
        tmp = tmp.reshape((-1))
        y[i] = tmp"""

    y = scaleTargets(y)
    #plotTargets(y)
    #y = logTargets(y)
    #plotTargets(y)

    print(np.mean(y, axis=0))
    print(np.min(y, axis=0))
    print(np.max(y, axis=0))
    print(np.std(y, axis=0))

    return x, y

def myTrainTestSplit(x, y, testSize):
    trainSize = 1.0 - testSize
    x_val_back = x[:int(testSize * len(x))]
    #x_val_forward = x[int(trainSize * len(x)):]
    y_val_back = y[:int(testSize * len(y))]
    #y_val_forward = y[int(trainSize * len(y)):]
    x = x[int(testSize * len(x)):int(trainSize * len(x))]
    y = y[int(testSize * len(y)):int(trainSize * len(y))]
    x_train, x_val_mid, y_train, y_val_mid = train_test_split(x, y, test_size=testSize, random_state=45, shuffle=True)

    #x_val = np.vstack([x_val_back, x_val_mid, x_val_forward])
    x_val = np.vstack([x_val_back, x_val_mid])
    #y_val = np.vstack([y_val_back, y_val_mid, y_val_forward])
    y_val = np.vstack([y_val_back, y_val_mid])

    print("Train shape: {}".format(x_train.shape))
    print("Validation shape: {}".format(x_val.shape))
    print("Relative validation size: {:.2%}".format( x_val.shape[0] / x_train.shape[0] ))

    return x_train, x_val, y_train, y_val

def buildModel(xInputShape, yInputShape, topFunc = "linear", lr=0.001):
    model = ResnetBuilder.build_resnet_34((1, xInputShape[0], xInputShape[1]), yInputShape[0],
                                          outputActivation=topFunc)
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    model.summary()
    return model

addSymbols = ["AUDUSD_i"]
targetSymbol = "AUDUSD_i"
samplesDir = "../data/train_samples/"
timeframe = "M10"
agentTypes = [
    "opener",
    #"buyer",
    #"seller",
    #"opener"
]

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, HuberRegressor
from sklearn.manifold import TSNE
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

for agentType in agentTypes:
    x_train, y_train = loadDataSet(samplesDir, agentType, targetSymbol, timeframe)
    #x = x[int(0.5 * len(x)):]
    #y = y[int(0.5 * len(y)):]
    x_train, y_train = adjustDataSet(x_train, y_train, agentType=agentType)
    x_train, x_val, y_train, y_val = myTrainTestSplit(x_train, y_train, testSize=0.05)
    #x_train = x_train[:40000]
    #y_train = y_train[:40000]

    ####################################
    #x_tsne = x_train[:10000]
    #x_tsne = TSNE(n_jobs=10, verbose=1).fit_transform(x_tsne)
    #plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=1)
    #plt.show()
    ####################################

    model = CatBoostRegressor(n_estimators=1000,
                              max_depth=8,
                              objective="MultiRMSE",
                              #early_stopping_rounds=100,
                              verbose=1)
    model.fit(x_train, y_train)

    #model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5, weights="uniform", n_jobs=3), n_jobs=3)
    #model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=30, weights="distance", n_jobs=3), n_jobs=3)
    #model = MultiOutputRegressor(KernelRidge(kernel="rbf", alpha=1.0), n_jobs=3)
    #model = MultiOutputRegressor(LinearRegression(n_jobs=3), n_jobs=3)
    #model = MultiOutputRegressor(SGDRegressor(random_state=45), n_jobs=3)
    #model = MultiOutputRegressor(CatBoostRegressor(n_estimators=100, max_depth=8, verbose=1), n_jobs=3)
    #model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=8, n_jobs=3), n_jobs=3)
    #model = MultiOutputRegressor(LGBMRegressor(n_estimators=200, max_depth=8, n_jobs=3, verbose=1), n_jobs=3)
    #model = MultiOutputRegressor(Lasso(alpha=0.01), n_jobs=3)
    #model.fit(x_train, y_train)

    y_pred = model.predict(x_train)
    score = mean_squared_error(y_train, y_pred)
    print("Train mse: {}".format(score) )

    y_pred = model.predict(x_val)
    score = mean_squared_error(y_val, y_pred)
    print("Val mse: {}".format(score))

print("done")