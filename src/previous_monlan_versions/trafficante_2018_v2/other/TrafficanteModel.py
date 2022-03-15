from other.DataManager import DataManager
from DataAnalyzer import DataAnalyzer
from PairSettings import PairSettings

import numpy as np
import matplotlib.pyplot as plt


class TrafficanteModel:

    def __init__(self):

        self.dataManager = DataManager()
        self.dataAnalyzer = DataAnalyzer()
        self.pairSettings = PairSettings()

        pass

    def trainModel(self, saveModel = True):

        settings = self.pairSettings
        dataManager = self.dataManager

        for i in range(settings.targetColumnsCount):

            barData = dataManager.readBarData(settings.pairName, settings.pairPeriod)
            domainColumns = settings.domainColumns
            targetColumn  = [settings.targetColumns[i]]
            dataManager.barDataScalers = dataManager.getScalersForBarData(barData, domainColumns)
            dataManager.indOfTargetSeq = dataManager.getIndexOfTargetColumn(barData, domainColumns, targetColumn)

            dataAnalyzer = self.dataAnalyzer
            dataAnalyzer.initWorkParams(len(domainColumns), settings.L, settings.P)

            print("Creating XY datasets for target sequence: " + targetColumn[0])
            (X_train, Y_train), (X_test, Y_test) = dataManager.makeDataSetsFromBarData(barData,
                                                                            settings.split_coef,
                                                                            domainColumns=domainColumns,
                                                                            targetSeq=targetColumn,
                                                                            modelIOParams=settings.workDataParams
                                                                            )
            print("Scaling XY train sets")
            (X_train, Y_train) = dataManager.scaleXYDataSets(X_train, Y_train, True)

            print("Scaling XY test sets")
            (X_test, Y_test) = dataManager.scaleXYDataSets(X_test, Y_test, True)

            print("Building model for " + targetColumn[0])
            dataAnalyzer.buildCRNN()
            dataAnalyzer.trainModel(X_train, Y_train, X_test, Y_test, epochs=settings.epochs, batch_size=settings.batch_size)

            if (saveModel == True):

                targetPostfix = targetColumn[0]
                targetPostfix = targetPostfix[1:]
                targetPostfix = targetPostfix[:-1]

                modelName = settings.pairName + "_" + settings.pairPeriod + "_" + targetPostfix
                print("Saving model " + modelName)
                dataAnalyzer.saveModel(modelName)

        pass

    def getPredictPlotData(self):

        dataManager = self.dataManager
        dataAnalyzer = self.dataAnalyzer
        settings = self.pairSettings

        wholePlotData = []

        for i in range(settings.targetColumnsCount):

            barData = dataManager.readBarData(settings.pairName, settings.pairPeriod)
            domainColumns = settings.domainColumns
            targetColumn  = [settings.targetColumns[i]]
            dataManager.barDataScalers = dataManager.getScalersForBarData(barData, domainColumns)
            dataManager.indOfTargetSeq = dataManager.getIndexOfTargetColumn(barData, domainColumns, targetColumn)

            targetPostfix = targetColumn[0]
            targetPostfix = targetPostfix[1:]
            targetPostfix = targetPostfix[:-1]

            modelName = settings.pairName + "_" + settings.pairPeriod + "_" + targetPostfix
            loadFlag = dataAnalyzer.loadModel(modelName)

            if loadFlag == False:
                print( "Model " + modelName + "doesn't exist. Creating and training model with current settings...." )
                self.trainModel()
                print( "Training complete. Loading model again. " )
                dataAnalyzer.loadModel( modelName )

            workParams = dataAnalyzer.extractIOParamsFromModel()
            dataAnalyzer.setWorkDataParams(workParams)

            ##########
            #workParams = settings.workDataParams
            ##########

            X_testSet = dataManager.getSetToPredict(barData, domainColumns, workParams)
            X_testSet = dataManager.scalePredictDataSet(X_testSet)
            Y_predict = dataAnalyzer.predict(X_testSet)

            N = workParams["inputSeqCount"]
            L = workParams["inputSeqLength"]
            P = workParams["predictSeqLength"]

            X_set = np.zeros((X_testSet.shape[1] - 1, ))
            for k in range(X_set.shape[0]):
                X_set[k] = X_testSet[dataManager.indOfTargetSeq][k+1][L-1]

            X_set = np.reshape(X_set, (1, X_set.shape[0]))
            X_set = dataManager.barDataScalers[dataManager.indOfTargetSeq].inverse_transform(X_set)

            Y_predict = np.reshape(Y_predict, (Y_predict.shape[1], Y_predict.shape[0]))
            Y_predict = dataManager.barDataScalers[dataManager.indOfTargetSeq].inverse_transform(Y_predict)

            plotData = {}
            plotData.update( {targetColumn[i] + "_real" : X_set.copy()} )
            plotData.update( {targetColumn[i] + "_forecast" : Y_predict.copy()} )

            wholePlotData.append(plotData)

        return wholePlotData

    def setSettings(self, pairSettings):

        self.pairSettings = pairSettings

    pass

    def updateBarData(self, updateFilePath):

        self.dataManager.updateBarData(updateFilePath, self.pairSettings.pairName, self.pairSettings.pairPeriod)



