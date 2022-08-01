
from keras.backend import *
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras import regularizers
from keras.callbacks import *
from keras.losses import *

import numpy as np
import pandas as pd
from scipy.spatial.distance import sqeuclidean
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from DataManager import DataManager
from PairSettings import PairSettings

import matplotlib.pyplot as plt

class DataAnalyzer:

    def __init__(self):

        self.model = None
        self.modelsDir = "./data/models/"

        self.workDataParams = {"inputSeqCount" : None, "inputSeqLength" : None, "predictSeqLength" : None}

        pass

    def buildCRNN(self):

        N = self.workDataParams["inputSeqCount"]
        L = self.workDataParams["inputSeqLength"]
        P = self.workDataParams["predictSeqLength"]

        input_size = (N, L)

        inputs = [Input(shape=(L, 1), name="input_" + str(i)) for i in range(0, N)]

        #mergedLayers = concatenate(inputs)

        gruLayers_1 = []
        for i in range (0, N):
            gruLayer = CuDNNLSTM(L,
                                return_sequences=True
                                )(inputs[i])
            gruLayers_1.append(Dropout(0.01)(gruLayer))

        mergedPools = concatenate(gruLayers_1)

        gruLayers_2 = CuDNNLSTM(N*L,
                                 return_sequences=False,
                             )(Dropout(0.01)(mergedPools))

        denseLayer_3 = Dense(P,
                           activation="linear",
                           )(gruLayers_2)

        model = Model(input = inputs, output = denseLayer_3)
        model.compile(optimizer=RMSprop(), loss="mean_squared_error", metrics=["mean_absolute_percentage_error"])
        model.summary()

        self.model = model

        return model

    def trainModel(self, x_train, y_train, x_test, y_test, epochs=200, batch_size=50):

        validInput = self.makeValidToInputDataset(x_train)

        model_checkpoint = ModelCheckpoint('analyzer_CRNN_checkpoint.hdf5', monitor='loss',verbose=1, save_best_only=True)
        self.model.fit(validInput, y_train, epochs=epochs, batch_size=batch_size, callbacks=[model_checkpoint], verbose=1)

        return self.model

    def saveModel(self, modelName):
        modelPath = self.modelsDir + modelName + ".hd5"
        self.model.save(modelPath)
        pass

    def loadModel(self, modelName):
        modelPath = self.modelsDir + modelName + ".hd5"

        if os.path.exists( modelPath ) == False:
            return False

        self.model = load_model(modelPath)
        self.model.summary()

        return True

    def predict(self, X_dataSet):

        validInput = self.makeValidToInputDataset(X_dataSet)

        predictedVals = self.model.predict(validInput, verbose=1)

        return predictedVals

    def makeValidToInputDataset(self, X_dataSet):

        N = self.workDataParams["inputSeqCount"]

        inputNameList = []
        for i in range(N):
            inputNameList.append("input_" + str(i))

        validInputDataSet = { }
        for i in range(N):
            arr = np.zeros(shape=(X_dataSet.shape[1], X_dataSet.shape[2]))
            for j in range(arr.shape[0]):
                for k in range(arr.shape[1]):
                    arr[j][k] = X_dataSet[i][j][k]
            #arr = arr.T
            arr = np.expand_dims(arr, axis=2)
            validInputDataSet.update({inputNameList[i] : arr.copy()})

        return validInputDataSet

    def initWorkParams(self, inputSeqCount, inputSeqLength, predictSeqLength):

        self.workDataParams["inputSeqCount"] = inputSeqCount
        self.workDataParams["inputSeqLength"] = inputSeqLength
        self.workDataParams["predictSeqLength"] = predictSeqLength

        pass

    def setWorkDataParams(self, workParams= {"inputSeqCount" : None, "inputSeqLength" : None, "predictSeqLength" : None}):

        self.workDataParams["inputSeqCount"] = workParams["inputSeqCount"]
        self.workDataParams["inputSeqLength"] = workParams["inputSeqLength"]
        self.workDataParams["predictSeqLength"] = workParams["predictSeqLength"]

        pass

    def getWorkDataParams(self):

        return self.workDataParams

    def extractIOParamsFromModel(self):

        inputSeqCount = 0
        for layer in self.model.layers:
            if ( list(layer.get_config().keys())[0] == 'batch_input_shape'):
                inputSeqCount += 1

        inputSeqLength = 0
        inputSeqLength = self.model.layers[0].get_config()["batch_input_shape"][1]

        lastLayerInd = len(self.model.layers) - 1
        predictSeqLength = self.model.layers[lastLayerInd].get_config()["units"]

        extractedParams = {}
        extractedParams["inputSeqCount"] = inputSeqCount
        extractedParams["inputSeqLength"] = inputSeqLength
        extractedParams["predictSeqLength"] = predictSeqLength


        return extractedParams

    def getPredictPlotData(self, X_data, Y_data):

        predictValsCount = Y_data.shape[0] + Y_data.shape[1] - 1

        x_axis = np.zeros((predictValsCount, ))
        for i in range(predictValsCount):
            x_axis[i] = i

        y_vals = np.zeros((predictValsCount, ))

        L = Y_data.shape[0]
        P = Y_data.shape[1]

        j = 0
        end = ( L // P)
        while j < end:
            for k in range(P):
                y_vals[j*P + k] = Y_data[j*P][k]
            j += 1

        newEnd = L - end * P
        for z in range(newEnd):
            y_vals[end * P + z] = Y_data[end * P][z]

        for j in range(P):
            y_vals[L - 1 + j] = Y_data[L - 1][j]

        return x_axis, y_vals

    def convertXYsetsToValidPlotData(self, X_real, Y_predict):

        N = self.workDataParams["inputSeqCount"]
        L = self.workDataParams["inputSeqLength"]
        P = self.workDataParams["predictSeqLength"]

        #find target sequence index
        TSI = None
        minDistance = float("Inf")

        for i in range (N):
            curDistance = 0

            x_arr = np.zeros((X_real.shape[1] - L, 1))
            y_arr = np.zeros((X_real.shape[1] - L, 1))

            for j in range(x_arr.shape[0]):
                x_arr[j] = X_real[i][j + L][0]
                y_arr[j] = Y_predict[j][0]

            curDistance = sqeuclidean(x_arr, y_arr)

            if (curDistance < minDistance):
                minDistance = curDistance
                TSI = i

        y_arr = np.zeros((Y_predict.shape[0], L + 1))

        for i in range( y_arr.shape[0] ):
            for j in range( L + 1 ):
                y_arr[i][j] = Y_predict[i][0]

        x_arr = np.zeros((X_real.shape[1], L + 1) )
        for i in range( x_arr.shape[0] - 1):
            for j in range( L ): #last val is zero especialy
                x_arr[i][j] = X_real[TSI][i + 1][j]


        y_arr = np.expand_dims(y_arr, axis=2)
        x_arr = np.expand_dims(x_arr, axis=2)

        inputNameList = []
        for i in range(2):
            inputNameList.append("input_" + str(i))

        plotData = {}
        plotData.update({inputNameList[0] : y_arr.copy()})
        plotData.update({inputNameList[1] : x_arr.copy()})


        return plotData
