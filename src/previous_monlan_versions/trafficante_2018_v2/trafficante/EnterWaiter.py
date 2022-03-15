
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from trafficante.CloserWaiter import CloserWaiter
from trafficante.Environment import Environment
from sklearn.preprocessing import MinMaxScaler

import numpy as np

class EnterWaiter:

    def __init__(self, cwStSize):

        self.model = None
        self.stSize = None
        self.modelsDir = "../data/models/"
        self.buyerWaiter = CloserWaiter( dealTypes="buy" )
        self.buyerWaiter.loadModel(cwStSize)
        self.sellerWaiter = CloserWaiter( dealTypes="sell" )
        self.sellerWaiter.loadModel(cwStSize)

        #one model for value func
        #separate models for value of buy, wait, sell

        pass

    def buildModel(self, stSize):
        inputLayer = Input(shape=(stSize, ))
        dense_1 = Dense(stSize * 2, activation="tanh")(inputLayer)
        dense_2 = Dense(stSize * 4, activation="tanh")(Dropout(0.00)(dense_1))
        dense_3 = Dense(stSize * 5, activation="tanh")(Dropout(0.00)(dense_2))
        dense_4 = Dense(stSize * 4, activation="tanh")(Dropout(0.00)(dense_3))
        dense_5 = Dense(stSize * 2, activation="tanh")(Dropout(0.00)(dense_4))
        outputLayer = Dense(3, activation="linear")(dense_5)

        model = Model(inputs=inputLayer, outputs=outputLayer)
        model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["mean_absolute_percentage_error"])
        model.summary()

        self.model = model
        return model

    def saveModel(self):
        self.model.save( self.modelsDir + "enter_waiter_{}.model".format( self.stSize ) )
        pass

    def loadModel(self, stSize):
        self.model = load_model( self.modelsDir + "enter_waiter_{}.model".format( self.stSize ) )
        pass

    def getAction(self, st):
        #choose max value action
        at_values = self.model.predict(st)
        at = np.argmax( at_values )
        return at

    def train(self, epochs = 10, stSize = 100):

        #buy or sell
        #random buy sell
        self.model = self.buildModel(stSize)
        self.stSize = stSize
        model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='loss',verbose=1, save_best_only=True)

        #env.setTimeLimits("12.10.2017", "20.11.2018")
        for i in range(epochs):

            env = Environment(stSize)
            env.setBarData("EURUSD", "H1")

            doneFlag = False
            while doneFlag == False:
                trainSt, trainAtValue, doneFlag, profits = self.getNextTrainSample(env)
                if (doneFlag == True):
                    break
                print( trainAtValue )
                predictedVals = self.model.predict(trainSt)
                print( predictedVals )
                print("||||||||||||||||||")
                self.model.fit(trainSt, trainAtValue, epochs=1, batch_size=trainSt.shape[0], callbacks=[model_checkpoint], verbose=0)
                predictedVals = self.model.predict(trainSt)
                print( predictedVals )
                print("profits: ", end="")
                print(profits)
                print("******************")
                print()
            print( "epoch {} completed".format( i ) )

        pass

    def scoreModel(self):

        pass

    def getNextTrainSample(self, env):

        startPosition = env.getStartInd()
        trainSt, _, globalDoneFlag = env.step()
        trainSt = np.asarray( trainSt ).reshape((-1, 1))
        trainSt = MinMaxScaler((-1, 1)).fit_transform( trainSt )
        trainSt = trainSt.reshape((1, -1))
        if (globalDoneFlag == True):
            return None, None, globalDoneFlag
        env.resetByIndex(startPosition)

        #get base price
        basePrice = env.getBasePrice()

        #trade buy, if close get total reward
        buyProfit = None
        buyStepCount = 0
        closeAction = False
        while closeAction == False:
            st, yt, doneFlag = env.step()
            globalDoneFlag = doneFlag
            if (globalDoneFlag == True):
                return None, None, globalDoneFlag
            buyStepCount += 1
            st = st - basePrice
            st = np.asarray( st ).reshape((-1, 1))
            st = MinMaxScaler((-1, 1)).fit_transform(st)
            st = st.reshape((1, -1))
            at = self.buyerWaiter.getAction(st)
            if at == 0: #close deal
                closeAction = True
                buyProfit = (yt - basePrice) / basePrice
        #reset env
        startPosition = env.getStartInd()
        startPosition -= buyStepCount
        env.resetByIndex(startPosition)


        #trade sell, if close get total reward
        sellProfit = None
        sellStepCount = 0
        closeAction = False
        while closeAction == False:
            st, yt, doneFlag = env.step()
            globalDoneFlag = doneFlag
            if (globalDoneFlag == True):
                return None, None, globalDoneFlag
            sellStepCount += 1
            st = -(st - basePrice) #sell profits in the past
            st = np.asarray( st ).reshape((-1, 1))
            st = MinMaxScaler((-1, 1)).fit_transform(st)
            st = st.reshape((1, -1))
            at = self.sellerWaiter.getAction(st)
            if at == 0: #close deal
                closeAction = True
                sellProfit = (-(yt - basePrice)) / basePrice
        #reset env
        startPosition = env.getStartInd()
        startPosition -= sellStepCount
        env.resetByIndex(startPosition)

        #set env pos state to the best deal moment
        startPosition = env.getStartInd()
        if (buyProfit <= 0.0 and sellProfit <= 0.0):
            startPosition += 1 #action is wait for the next timeframe
        elif ( buyProfit >= sellProfit):
            startPosition += buyStepCount
        else:
            startPosition += sellStepCount
        env.resetByIndex( startPosition )

        trainAtValues = np.zeros( (1, 3) )
        if (buyProfit <= 0.0 and sellProfit <= 0.0):
            trainAtValues[0][0] = -1 #buy
            trainAtValues[0][1] = 1  #wait
            trainAtValues[0][2] = -1 #sell
        elif ( buyProfit >= sellProfit):
            trainAtValues[0][0] = 1 #buy
            trainAtValues[0][1] = -1  #wait
            trainAtValues[0][2] = -1 #sell
        else:
            trainAtValues[0][0] = -1 #buy
            trainAtValues[0][1] = -1  #wait
            trainAtValues[0][2] = 1 #sell

        profits = []
        profits.append(buyProfit)
        profits.append( 0 )
        profits.append(sellProfit)

        return trainSt, trainAtValues, globalDoneFlag, profits