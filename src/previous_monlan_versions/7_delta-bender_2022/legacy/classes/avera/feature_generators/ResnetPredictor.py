'''
Created on Dec 31, 2019

@author: bookknight
'''


from avera.agents.ResnetBuilder import ResnetBuilder
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import joblib

class ResnetPredictor():

    def __init__(self):
        self.model = None
        self.savedInputShape = None
        self.savedOutputShape = None
        self.savedLr = None
        pass

    def build_model(self, inputShape, outputShape, lr):
        model = ResnetBuilder.build_resnet_34((1, inputShape[0], inputShape[1]), outputShape[0], outputActivation="linear")
        # model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=lr))
        self.model = model
        self.savedInputShape = inputShape
        self.savedOutputShape = outputShape
        self.savedLr = lr
        pass

    def fit(self, x, y, validation_data=None, batch_size=32, epochs=10, verbose=1):
        if validation_data is None:
            self.model.fit(x, y, batch_size=batch_size, verbose=verbose, epochs=epochs, shuffle=True)
        else:
            self.model.fit(x, y, validation_data=validation_data,
                           batch_size=batch_size, verbose=verbose, epochs=epochs, shuffle=True)
        return self

    def predict(self, x):
        predicts = self.model.predict(x, batch_size=32)
        return predicts

    def score(self, testX, testY):
        predicts = self.predict(testX)
        scores = []
        for i in range(len(predicts)):
            scores.append(np.sqrt(np.sum(np.square(predicts[i] - testY[i]))))
        score = np.mean(scores)
        return score

    #util functions
    def makeTrainTestDataSets(self, dfList, nDiffs, featureList, nPoints=32):

        modDfList = []
        iDf = 0
        for df in dfList:
            preprocDf = self.preprocDf(df, nDiffs, featureList)
            for colName in preprocDf.columns:
                if colName not in featureList:
                    del preprocDf[colName]
            modDfList.append(preprocDf)
            iDf += 1
            print("Preproc {} df of {}".format(iDf, len(dfList)))

        x = []
        y = []
        for df in modDfList:
            df = df.values
            for i in range(len(df) - nPoints):
                x.append( df[i:i+nPoints] )
                y.append( df[i+nPoints] )
        x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.05, random_state = 42)

        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        return x_train, x_test, y_train, y_test


    def preprocDf(self, df, nDiffs, featureList, returnScalers=False):
        dfCopy = df.copy()
        scalerDict = {}
        for feat in featureList:
            scalerDict[feat] = StandardScaler()

        dfCols = list(dfCopy.columns)
        for feat in featureList:
            if feat not in dfCols:
                raise ValueError("df doesn't contain column: \"{}\" ".format(feat))

        for feat in featureList:
            diffVals = dfCopy.copy()
            for i in range(nDiffs):
                notShifted = diffVals[feat]
                shiftedData = diffVals[feat].shift(periods=1)
                diffVals[feat] = notShifted - shiftedData
            tmp = diffVals[feat].values
            #tmp = tmp[1:]
            tmp = np.reshape(tmp, (-1, 1))
            scalerDict[feat].fit(tmp)

        for i in range(nDiffs):
            for feat in featureList:
                notShifted = dfCopy[feat]
                shiftedData = dfCopy[feat].shift(periods=1)
                dfCopy[feat] = notShifted - shiftedData
            iter = next(dfCopy.iterrows())
            dfCopy = dfCopy.drop(iter[0])

        for feat in featureList:
            data = dfCopy[feat].values
            data = data.reshape(-1, 1)
            data = scalerDict[feat].transform(data)
            dfCopy[feat] = data

        if returnScalers:
            return dfCopy, scalerDict
        else:
            return dfCopy

    def save(self, dir, name):
        self.model.save_weights(dir + name + "_weights" + ".h5")

        tmp_1 = self.model
        self.model = None
        import joblib
        with open(dir + name + ".pkl", mode="wb") as agentFile:
            joblib.dump(self, agentFile)
        self.model = tmp_1
        pass

    def load(self, dir, name):
        import joblib
        from avera.utils.keras_utils import reset_keras
        with open(dir + name + ".pkl", mode="rb") as agentFile:
            loadedAgent = joblib.load(agentFile)

        self.model = None
        #reset_keras()
        inputShape = loadedAgent.savedInputShape
        outputShape = loadedAgent.savedOutputShape
        lr = loadedAgent.savedLr
        loadedAgent.build_model(inputShape, outputShape, lr)
        loadedAgent.model.load_weights(dir + name + "_weights" + ".h5")
        return loadedAgent
