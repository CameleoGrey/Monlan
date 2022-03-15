
from DataManager import DataManager
from DataAnalyzer import DataAnalyzer
from PairSettings import PairSettings
from TrafficanteModel import TrafficanteModel

import numpy as np
import matplotlib.pyplot as plt
import copy

trafficantePredictor = TrafficanteModel()
trafficantePredictor.trainModel( saveModel=True )

#################################
#Test model
#################################

dataManager = DataManager()
dataAnalyzer = DataAnalyzer()
settings = PairSettings()

barData = dataManager.readBarData(settings.pairName, settings.pairPeriod)
domainColumns = settings.domainColumns
targetColumn  = [settings.targetColumns[0]]
dataManager.barDataScalers = dataManager.getScalersForBarData(barData, domainColumns)
dataManager.indOfTargetSeq = dataManager.getIndexOfTargetColumn(barData, domainColumns, targetColumn)

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

targetPostfix = targetColumn[0]
targetPostfix = targetPostfix[1:]
targetPostfix = targetPostfix[:-1]

modelName = settings.pairName + "_" + settings.pairPeriod + "_" + targetPostfix
dataAnalyzer.loadModel(modelName)
workParams = dataAnalyzer.extractIOParamsFromModel()
dataAnalyzer.setWorkDataParams(workParams)

###################################
#evaluate model
###################################

print( "Evaluating model..." )
validSet = dataAnalyzer.makeValidToInputDataset(X_test)
score = dataAnalyzer.model.evaluate(validSet, Y_test, batch_size = 10, verbose = 1)
print( "Test loss: ", score[0] )
print( "Test accuracy: ", score[1] )

##################################################################
plt.subplot(311)

Y_predicted = dataAnalyzer.predict(X_train)

x_train, y_train = dataAnalyzer.getPredictPlotData(X_train, Y_train)
plt.plot(x_train, y_train)

x_train, y_trainPredict = dataAnalyzer.getPredictPlotData(X_train, Y_predicted)
plt.plot(x_train, y_trainPredict)

##########################################################
plt.subplot(312)

Y_predicted = dataAnalyzer.predict(X_test)

x_test, y_test = dataAnalyzer.getPredictPlotData(X_test, Y_test)
plt.plot(x_test, y_test)

x_test, y_testPredict = dataAnalyzer.getPredictPlotData(X_test, Y_predicted)
plt.plot(x_test, y_testPredict)


###########################################################################
plt.subplot(313)
X_set = dataManager.getSetToPredict(barData, domainColumns, workParams)
X_set = dataManager.scalePredictDataSet(X_set)
L = X_set.shape[1]
P = X_set.shape[2]

x_history = np.zeros((L-1, ))
for i in range(L-1):
    x_history[i] = i

y_history = np.zeros((L-1, ))
for i in range(L-1):
    y_history[i] = X_set[0][i+1][P-1]

plt.plot(x_history, y_history)

##################################################################################

Y_predict = dataAnalyzer.predict(X_set)

x_realPredict, y_realPredict = dataAnalyzer.getPredictPlotData(X_set, Y_predict)

plt.plot(x_realPredict, y_realPredict)

plt.show()
