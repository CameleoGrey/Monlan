import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm
import pandas as pd
import uuid
from datetime import datetime

class RiskManager:
    def __init__(self, nPoints, nLevels, stopPos, takePos, spreadCoef):

        self.nPoints = nPoints
        self.nLevels = nLevels
        self.stopPos = stopPos
        self.takePos = takePos
        self.spreadCoef = spreadCoef

        pass

    def getAdaptiveLimits(self, datetimeStr, historyData, dealOpenPrice):

        obsList = historyData.loc[:str(datetimeStr)].tail(self.nPoints).copy()
        avgVals, levelLabels = self.findLevels(obsList, self.nPoints, self.nLevels)
        uniqLevels = np.sort(np.unique(levelLabels))
        avgLevels = np.zeros((uniqLevels.shape[0],))
        for i in range(uniqLevels.shape[0]):
            avgLevels[i] = np.average( avgVals[ levelLabels == uniqLevels[i] ] )
        dists = pairwise_distances(np.array(dealOpenPrice).reshape((-1, 1)), avgLevels.reshape((-1, 1)), metric="euclidean", n_jobs=1)
        dists = dists[0]

        sortedDists = np.sort(dists)
        limitsDict = {}
        limitsDict["stop"] = int(sortedDists[self.stopPos] / self.spreadCoef)
        limitsDict["take"] = int(sortedDists[self.takePos] / self.spreadCoef)

        return limitsDict

    def findLevels(self, df, nPoints = 110, nLevels=4):
        avgVals = df.tail(nPoints)
        avgVals = avgVals[["open", "close"]]
        avgVals = (avgVals["open"].values + avgVals["close"].values) / 2.0

        avgVals = avgVals.reshape((-1, 1))
        pointDists = pairwise_distances(avgVals, avgVals, metric="euclidean", n_jobs=1)
        avgVals = avgVals.reshape((-1,))

        pointDists = MinMaxScaler().fit_transform(pointDists)

        #startTime = datetime.now()
        clusterizer = KMeans(n_clusters=nLevels,
                             random_state=45,
                             init="random",
                             algorithm="elkan",
                             n_init=1).fit(pointDists)
        levels = clusterizer.labels_
        #endTime = datetime.now()
        #print("clust: " + str(endTime - startTime))

        """stubX = np.linspace(0, 1, avgVals.shape[0])
        plt.scatter(stubX, avgVals, c=levels, s=5)
        uniqLevels = np.sort(np.unique(levels))
        for unLev in uniqLevels:
            plotLevelVals = avgVals[levels == unLev]
            levelVal = np.average(plotLevelVals)
            # print(levelVals.shape)
            plt.axhline(y=levelVal, c="black", linestyle="-")
        plt.show()
        #plt.savefig("./images/{}.png".format(str(uuid.uuid4())), dpi=300)
        #plt.clf()"""

        return avgVals, levels


