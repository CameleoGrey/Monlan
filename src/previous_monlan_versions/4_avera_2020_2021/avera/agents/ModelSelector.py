

from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam
from keras.models import Sequential

import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import spearmanr, pearsonr

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import SGDRegressor, TheilSenRegressor, HuberRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
import glob



def useCpu(nThreads, nCores):
    import tensorflow as tf
    from keras import backend as K
    numThreads = nThreads
    num_CPU = nCores
    num_GPU = 0
    config = tf.ConfigProto(intra_op_parallelism_threads=numThreads,
                            inter_op_parallelism_threads=numThreads,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU}
                            )
    session = tf.Session(config=config)
    K.set_session(session)

useCpu(nThreads=8, nCores=8)

class MySelector():
    def __init__(self, featShape):
        self.model = self.buildFlatModel(featShape)
        pass

    def buildFlatModel(self, featShape):
        model = Sequential()
        model.add(Dense(64, input_dim=featShape, activation='elu', kernel_initializer='glorot_uniform'))
        model.add(Dense(128, activation='elu', kernel_initializer='glorot_uniform', kernel_regularizer=l1_l2(0.001, 0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='elu', kernel_initializer='glorot_uniform', kernel_regularizer=l1_l2(0.001, 0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='elu', kernel_initializer='glorot_uniform', kernel_regularizer=l1_l2(0.001, 0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='elu', kernel_initializer='glorot_uniform', kernel_regularizer=l1_l2(0.001, 0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='elu', kernel_initializer='glorot_uniform', kernel_regularizer=l1_l2(0.001, 0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='elu', kernel_initializer='glorot_uniform', kernel_regularizer=l1_l2(0.001, 0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='elu', kernel_initializer='glorot_uniform'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        return model

    def fit(self, X, Y):
        self.model.fit(X, Y, batch_size=200, epochs=10000, verbose=1)
        pass

    def predict(self, X):
        preds = self.model.predict(X)
        preds = np.reshape(preds, newshape=(-1, ))
        return preds

class ModelSelector():
    def __init__(self):

        self.modelEstimator = CatBoostRegressor(iterations=2000)
        #self.modelEstimator = CatBoostClassifier(iterations=2000)
        #self.modelEstimator = MySelector(featShape=25)
        #self.modelEstimator = TheilSenRegressor()
        #self.modelEstimator = KNeighborsClassifier(n_neighbors=20)
        #self.modelEstimator = SVR()
        #self.modelEstimator = SGDRegressor()
        #self.modelEstimator = HuberRegressor()

        pass

    #############################################
    # data processing
    #############################################

    def loadExperimentData(self, symbol):

        dealsPacks = []
        testDealsPaths = glob.glob("./testDealsStatistics_{}_*.pkl".format(symbol))
        backDealsPaths = glob.glob("./backDealsStatistics_{}_*.pkl".format(symbol))
        trainDealsPaths = glob.glob("./trainDealsStatistics_{}_*.pkl".format(symbol))

        for i in range(len(testDealsPaths)):
            testDealsStatistics = None
            backDealsStatistics = None
            trainDealsStatistics = None
            with open(testDealsPaths[i], mode="rb") as dealsFile:
                testDealsStatistics = joblib.load(dealsFile)
            with open(backDealsPaths[i], mode="rb") as dealsFile:
                backDealsStatistics = joblib.load(dealsFile)
            with open(trainDealsPaths[i], mode="rb") as dealsFile:
                trainDealsStatistics = joblib.load(dealsFile)
            dealsStatistics = {}
            dealsStatistics["test"] = testDealsStatistics
            dealsStatistics["train"] = trainDealsStatistics
            dealsStatistics["back"] = backDealsStatistics
            dealsPacks.append(dealsStatistics)

        return dealsPacks

    def buildDataSet(self, experimentData, normalizeSet = False):

        episodesCount = len(experimentData["test"])

        X = []
        Y = []
        for i in range(episodesCount):
            trainDeals = experimentData["train"][i]
            testDeals = experimentData["test"][i]
            backDeals = experimentData["back"][i]

            #ignore outliers
            if len(trainDeals) < 3 or len(testDeals) < 3 or len(backDeals) < 3:
                continue

            sampleX, sampleY = self.buildSample( i, trainDeals, testDeals, backDeals )
            X.append( sampleX )
            Y.append( sampleY )

        if normalizeSet:
            X, Y = self.normalizeDataSet(X, Y)

        return [X, Y]

    def buildSample(self, trainEp, trainDeals, testDeals, backDeals):

        sampleX = self.extractFeatures( trainEp, trainDeals, backDeals)

        sampleY = self.getCumulativeReward(testDeals)[-1]

        #if sampleY > 0:
        #    sampleY = 1
        #else:
        #    sampleY = -1


        return sampleX, sampleY

    def extractFeatures(self, trainEp, trainDeals, backDeals):

        features = []

        features.append( trainEp )

        features.append( len(trainDeals) )
        features.append( len(backDeals) )

        trainCumRew = self.getCumulativeReward(trainDeals)
        backCumRew = self.getCumulativeReward(backDeals)
        features.append( trainCumRew[-1] )
        features.append( backCumRew[-1] )
        features.append( np.mean(trainCumRew) )
        features.append( np.mean(backCumRew) )
        features.append(np.median(trainCumRew))
        features.append(np.median(backCumRew))
        features.append(np.std(trainCumRew))
        features.append(np.std(backCumRew))

        features.append(np.mean(trainDeals))
        features.append(np.mean(backDeals))
        features.append(np.median(trainDeals))
        features.append(np.median(backDeals))
        features.append(np.std(trainDeals))
        features.append(np.std(backDeals))

        features.append(np.max(trainDeals))
        features.append(np.max(backDeals))
        features.append(np.min(trainDeals))
        features.append(np.min(backDeals))

        features.append(np.max(trainDeals) - np.min(trainDeals))
        features.append(np.max(trainDeals) - np.min(backDeals))

        features.append( np.max(trainDeals) - np.max(backDeals) )
        features.append( np.min(trainDeals) - np.min(backDeals) )

        """linTrain = self.convertToLinearSet(trainCumRew)
        linBack = self.convertToLinearSet(backCumRew)
        trainBackCorrCoef = self.getCorrCoef(linTrain, linBack)
        features.append(trainBackCorrCoef)

        synteticFeats = []
        #for i in range(len(features)):
        #    for j in range(len(features)):
        #        synteticFeats.append( features[i] * features[j] )
        #for i in range(len(features)):
        #    synteticFeats.append( np.sin( features[i] ))
        #for i in range(len(features)):
        #    synteticFeats.append( np.cos( features[i] ))
        #for i in range(len(features)):
        #    synteticFeats.append( np.sin(features[i]) * np.cos(features[i]))
        #for i in range(len(features)):
        #    synteticFeats.append( features[i]**2 )
        #for i in range(len(features)):
        #    synteticFeats.append( features[i]**3 )


        for i in range(len(synteticFeats)):
            features.append(synteticFeats[i])"""


        return features

    def normalizeDataSet(self, X, Y):

        for j in range(len(X[0])):
            valsToNorm = []
            for i in range(len(X)):
                valsToNorm.append(X[i][j])
            valsToNorm = np.reshape( valsToNorm, newshape=(-1, 1))
            normVals = MinMaxScaler(feature_range=(-1, 1)).fit_transform(valsToNorm)
            #normVals = StandardScaler().fit_transform(valsToNorm)
            normVals = np.reshape(normVals, newshape=(-1, ))
            for i in range(len(normVals)):
                X[i][j] = normVals[i]

        #Y = np.reshape( Y, newshape=(-1, 1))
        #Y = MinMaxScaler(feature_range=(-1, 1)).fit_transform(Y)
        #Y = StandardScaler().fit_transform(Y)
        #Y = np.reshape(Y, newshape=(-1, ))

        return X, Y

    ############################################
    # feature extraction
    ############################################
    def getCumulativeReward(self, deals):
        sumRew = 0
        cumulativeReward = []
        for i in range(len(deals)):
            sumRew += deals[i]
            cumulativeReward.append(sumRew)
        return cumulativeReward

    def convertToLinearSet(self, dealsStatistics):
        #import matplotlib.pyplot as plt
        dsCopy = dealsStatistics.copy()
        dsCopy = np.reshape(dsCopy, (-1,))
        X = np.array([x for x in range(len(dsCopy))])
        X = np.reshape(X, (-1, 1))
        linReg = HuberRegressor().fit(X, dsCopy)
        tmp = linReg.predict(X)
        #plt.plot(X, tmp)
        #plt.plot(X, dealsStatistics)
        #plt.show()
        return tmp

    def getCorrCoef(self, trainCumRew, backCumRew):

        minLen = min([len(backCumRew)])
        tmpTrain = np.sort(np.random.choice(trainCumRew, size=minLen))
        tmpBack = np.sort(np.random.choice(backCumRew, size=minLen))

        #trainBackCorr = spearmanr(tmpTrain, tmpBack).correlation
        trainBackCorr = pearsonr(tmpTrain, tmpBack)[0]

        return trainBackCorr


    ############################################
    # working
    ############################################
    def fit(self, X, Y, evalSet=None):

        if evalSet is None:
            self.modelEstimator.fit(X, Y)
        else:
            self.modelEstimator.fit(X, Y, eval_set=evalSet)

        pass

    """
    x - deals statistics list of trained agent 
    """
    def predict(self, X):

        preds = self.modelEstimator.predict(X)

        return preds
