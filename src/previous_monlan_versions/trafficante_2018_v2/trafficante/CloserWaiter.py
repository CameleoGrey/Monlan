
from keras.models import *
from keras.backend import *
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras import regularizers
from keras.callbacks import *
from keras.losses import *
from keras.models import load_model

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from trafficante.Environment import Environment

class CloserWaiter:

    #dealType: "buy", "sell"
    def __init__(self, dealTypes):

        #one model for value func
        #separate models for value of buy, wait, sell
        self.modelsDir = "../data/models/"
        self.model = None
        self.stSize = None
        self.dealTypes = dealTypes
        self.scaler = MinMaxScaler((-1, 1))

        pass

    def buildModel(self, stSize):

        """inputLayer = Input(shape=(stSize, ))
        dense_1 = Dense(stSize * 2, activation="tanh")(inputLayer)
        #dense_1 = BatchNormalization()(dense_1)
        dense_2 = Dense(stSize * 4, activation="tanh")(Dropout(0.05)(dense_1))
        #dense_2 = BatchNormalization()(dense_2)
        dense_3 = Dense(stSize * 5, activation="tanh")(Dropout(0.05)(dense_2))"""

        inputLayer = Input(shape=(stSize, 1))
        #lstm_1 = CuDNNLSTM(stSize * 5, return_sequences=True)(inputLayer)

        conv_1 = Conv1D( stSize * 5, 10, padding="same", activation="relu" )(inputLayer)
        maxPool_1 = MaxPool1D( )(conv_1)
        #dense_1 = CuDNNLSTM(stSize * 2, return_sequences=True)(maxPool_1)
        conv_2 = Conv1D( stSize * 4, 5, padding="same", activation="relu" )(maxPool_1)
        maxPool_2 = MaxPool1D( )(conv_2)
        conv_3 = Conv1D( stSize * 3, 4, padding="same", activation="relu" )(maxPool_2)
        maxPool_3 = MaxPool1D( )(conv_3)
        conv_4 = Conv1D( stSize * 2, 3, padding="same", activation="relu" )(maxPool_3)
        maxPool_4 = MaxPool1D()(conv_4)

        #dense_3 = BatchNormalization()(dense_3)
        dense_4 = Dense(stSize * 4, activation="relu")((Flatten()(maxPool_4)))
        #dense_4 = BatchNormalization()(dense_4)
        dense_5 = Dense(stSize * 2, activation="relu")(Dropout(0.05)(dense_4))
        #dense_5 = BatchNormalization()(dense_5)
        outputLayer = Dense(2, activation="linear")(dense_5)
        #outputLayer = BatchNormalization()(outputLayer)

        """inputLayer = Input(shape=(stSize, 1))
        conv_1 = Conv1D( stSize * 5, 2, padding="same", activation="tanh" )(inputLayer)
        maxPool_1 = MaxPool1D( )(conv_1)
        #dense_1 = CuDNNLSTM(stSize * 2, return_sequences=True)(maxPool_1)
        conv_2 = Conv1D( stSize * 5, 2, padding="same", activation="tanh" )(maxPool_1)
        maxPool_2 = MaxPool1D( )(conv_2)
        conv_3 = Conv1D( stSize * 5, 2, padding="same", activation="tanh" )(maxPool_2)
        maxPool_3 = MaxPool1D( )(conv_3)
        conv_4 = Conv1D( stSize * 5, 2, padding="same", activation="tanh" )(maxPool_3)
        maxPool_4 = MaxPool1D()(conv_4)
        outputLayer = Dense(2, activation="linear")(Flatten()(maxPool_4))"""

        model = Model(inputs=inputLayer, outputs=outputLayer)
        #model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
        model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["accuracy"])
        model.summary()

        self.model = model
        return model

    def saveModel(self):
        self.model.save( self.modelsDir + "closer_waiter_{}_{}.model".format(self.dealTypes, self.stSize) )
        pass

    def loadModel(self, stSize):
        self.model = load_model( self.modelsDir + "closer_waiter_{}_{}.model".format(self.dealTypes, stSize) )
        self.stSize = stSize
        pass

    def saveScaler(self, pair, period):
        with open( "../data/models/scaler_{}_{}_{}.pkl".format(self.dealTypes, pair, period), "wb" ) as scalerFile:
            joblib.dump( self.scaler, scalerFile )
        pass

    def loadScaler(self, pair, period):
        with open( "../data/models/scaler_{}_{}_{}.pkl".format(self.dealTypes, pair, period), "rb" ) as scalerFile:
            self.scaler = joblib.load( scalerFile )
        pass

    def getAction(self, st):
        #choose max value action
        at_values = self.model.predict(st)
        at = np.argmax( at_values )
        return at

    def train(self, pair, period, epochs = 10, maxFrames = 6, stSize = 100):

        #buy or sell
        #random buy sell
        self.model = self.buildModel(stSize)
        self.stSize = stSize
        model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='loss',verbose=1, save_best_only=True)

        #env.setTimeLimits("12.10.2017", "20.11.2018")
        for i in range(epochs):

            env = Environment(stSize)
            env.setBarData(pair, period)

            doneFlag = False
            while doneFlag == False:
                trainSt, trainAtValue, doneFlag = self.getNextTrainSample(env, maxFrames)

                expandedTrainSet = np.expand_dims( trainSt, axis=2 )
                #expandedTrainSet = trainSt
                print( trainAtValue )
                predictedVals = self.model.predict(expandedTrainSet)
                print( predictedVals )
                print("||||||||||||||||||")
                self.model.fit(expandedTrainSet, trainAtValue, epochs=10, batch_size=20, callbacks=[model_checkpoint], verbose=0)
                predictedVals = self.model.predict(expandedTrainSet)
                print( predictedVals )
                print("******************")
                print()
            print( "epoch {} completed".format( i + 1) )

        pass

    def scoreModel(self):

        pass

    def getNextTrainSample(self, env, maxFramesCount):

        #base_price is open price
        basePrice = env.getBasePrice()
        states = []
        prices = []
        for i in range(maxFramesCount):
            st, yt, doneFlag = env.step()
            states.append(st)
            prices.append(yt)
            if doneFlag == True:
                break

        prices = np.asarray( prices )
        states = np.asarray( states )
        prices = prices - basePrice
        states = states - basePrice
        if (self.dealTypes == "sell"):
            prices = -prices
            states = -states

        bestPointInd = np.argmax( prices )

        bestStates = []
        actionsValues = np.zeros( (bestPointInd+1, 2) )
        for i in range( bestPointInd + 1 ):
            bestStates.append( states[i] )
            if i == bestPointInd:
                actionsValues[i][0] = bestPointInd+1
                actionsValues[i][1] = -(bestPointInd+1) / 2
            else:
                actionsValues[i][0] = -1
                actionsValues[i][1] = 1

        trainSt = np.asarray( bestStates )
        trainAtValues = actionsValues

        #reset env to the state after close action (base price starts here)
        startPosition = env.getStartInd()
        startPosition -= maxFramesCount - bestPointInd - 1
        env.resetByIndex(startPosition)

        for k in range( trainSt.shape[0] ):
            tmp = trainSt[k]
            tmp = tmp.reshape((-1, 1))
            self.scaler = self.scaler.fit( tmp )
            tmp = self.scaler.transform( tmp )
            tmp = tmp.reshape((1, -1))
            trainSt[k] = tmp[0]

        return trainSt, trainAtValues, doneFlag
