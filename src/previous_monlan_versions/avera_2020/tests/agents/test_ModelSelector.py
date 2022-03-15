
from avera.agents.ModelSelector import ModelSelector
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

modelSelector = ModelSelector()

symbolList = [ "USDCAD_i" ]

experimentDataPacks = []
for symbol in symbolList:
    #for i in range(nExperiments):
    loadedPack = modelSelector.loadExperimentData(symbol)
    for pack in loadedPack:
        experimentDataPacks.append(pack)

dataSets = []
for i in range(len(experimentDataPacks)):
    buildedSet = modelSelector.buildDataSet( experimentDataPacks[i], normalizeSet=False )
    dataSets.append(buildedSet)

X = []
Y = []
for i in range(len(dataSets)):
    X.append(dataSets[i][0])
    Y.append(dataSets[i][1])
X = np.vstack(X)
Y = np.hstack(Y)

#tsne = TSNE( n_components=2, perplexity=10, n_iter=2000, n_jobs=10)
#modX = tsne.fit_transform(X, Y)
#plt.scatter( modX[:,0], modX[:,1] )
##plt.scatter( modX, Y )
#plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
modelSelector.fit(X_train, Y_train)

preds = modelSelector.predict(X_test)

print(preds)
print(Y_test)
corrCoef = pearsonr(preds, Y_test)[0]
print( corrCoef )

signs = []
nullCount = 0
oneCount = 0
for i in range(len(preds)):
    tmp = preds[i] * Y_test[i]
    if preds[i] > 0 and Y_test[i] > 0:
        oneCount += 1
        signs.append(1)
    else:
        if preds[i] > 0 and Y_test[i] < 0:
            nullCount += 1
            signs.append(-1)
        else:
            signs.append(0)
print(signs)
print(nullCount)
print(oneCount)
print(oneCount / (nullCount + oneCount))

print(modelSelector.modelEstimator.get_feature_importance())
