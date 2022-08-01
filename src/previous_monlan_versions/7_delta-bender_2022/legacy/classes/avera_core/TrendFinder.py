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

class TrendFinder:
    def __init__(self):
        pass

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

    def getLevelFeatures(self, df, nPoints=110, nLevels=4):

        dataPoints, levels = self.findLevels(df, nPoints, nLevels)
        uniqLevels = np.sort(np.unique(levels))

        levelVals = []
        levelAngles = []
        #linRegr = HuberRegressor()
        linRegr = LinearRegression(n_jobs=8)
        #startTime = datetime.now()
        for unLev in uniqLevels:
            levelData = dataPoints[levels == unLev]
            levelVal = np.average(levelData)
            levelVals.append( levelVal )

            stubX = np.linspace(0, 1, levelData.shape[0], endpoint=True)
            stubX = np.reshape(stubX, newshape=(-1, 1))

            regrData = np.reshape(levelData, newshape=(-1, 1))
            regrData = MinMaxScaler().fit_transform(regrData)
            regrData = np.reshape(regrData, newshape=(-1))

            linRegr.fit(stubX, regrData)
            regX = np.array([0.0, 1.0]).reshape((-1, 1))
            regY = linRegr.predict(regX)
            levelAngle = self.getTrendAngle(regX[0][0], regY[0], regX[1][0], regY[1])
            levelAngles.append( levelAngle )
        #endTime = datetime.now()
        #print(str(endTime - startTime))


        """stubX = np.linspace(0, 1, dataPoints.shape[0])
        plt.scatter(stubX, dataPoints, c=levels, s=5)
        for unLev in uniqLevels:
            plotLevelVals = dataPoints[levels == unLev]
            levelVal = np.average(plotLevelVals)
            # print(levelVals.shape)
            plt.axhline(y=levelVal, c="black", linestyle="-")

            levelX = stubX[levels == unLev]
            levelX = np.reshape(levelX, (-1, 1))
            linRegr.fit(levelX, plotLevelVals)
            regX = np.array([levelX[0][0], levelX[-1][0]]).reshape((-1, 1))
            regY = linRegr.predict(regX)
            plt.plot([levelX[0][0], levelX[-1][0]], [regY[0], regY[1]])
        plt.show()
        #plt.savefig("./images/{}.png".format(str(uuid.uuid4())), dpi=300)
        plt.clf()"""

        levelStrengths = []
        for unLev in uniqLevels:
            levelData = dataPoints[levels == unLev]
            lvlStren = levelData.shape[0] / dataPoints.shape[0]
            levelStrengths.append(lvlStren)

        levelFeatures = levelVals + levelAngles + levelStrengths

        return levelFeatures

    def findVolumeLevels(self, df, nPoints = 110, nLevels=4):
        avgVals = df.tail(nPoints)
        avgVals = avgVals[["tick_volume"]].values

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

    def getVolumeFeatures(self, df, nPoints=110, nLevels=4):

        dataPoints, levels = self.findVolumeLevels(df, nPoints, nLevels)
        uniqLevels = np.sort(np.unique(levels))

        levelVals = []
        levelAngles = []
        #linRegr = HuberRegressor()
        linRegr = LinearRegression(n_jobs=8)
        #startTime = datetime.now()
        for unLev in uniqLevels:
            levelData = dataPoints[levels == unLev]
            levelVal = np.average(levelData)
            levelVals.append( levelVal )

            stubX = np.linspace(0, 1, levelData.shape[0], endpoint=True)
            stubX = np.reshape(stubX, newshape=(-1, 1))

            regrData = np.reshape(levelData, newshape=(-1, 1))
            regrData = MinMaxScaler().fit_transform(regrData)
            regrData = np.reshape(regrData, newshape=(-1))

            linRegr.fit(stubX, regrData)
            regX = np.array([0.0, 1.0]).reshape((-1, 1))
            regY = linRegr.predict(regX)
            levelAngle = self.getTrendAngle(regX[0][0], regY[0], regX[1][0], regY[1])
            levelAngles.append( levelAngle )
        #endTime = datetime.now()
        #print(str(endTime - startTime))


        """stubX = np.linspace(0, 1, dataPoints.shape[0])
        plt.scatter(stubX, dataPoints, c=levels, s=5)
        for unLev in uniqLevels:
            plotLevelVals = dataPoints[levels == unLev]
            levelVal = np.average(plotLevelVals)
            # print(levelVals.shape)
            plt.axhline(y=levelVal, c="black", linestyle="-")

            levelX = stubX[levels == unLev]
            levelX = np.reshape(levelX, (-1, 1))
            linRegr.fit(levelX, plotLevelVals)
            regX = np.array([levelX[0][0], levelX[-1][0]]).reshape((-1, 1))
            regY = linRegr.predict(regX)
            plt.plot([levelX[0][0], levelX[-1][0]], [regY[0], regY[1]])
        plt.show()
        #plt.savefig("./images/{}.png".format(str(uuid.uuid4())), dpi=300)
        plt.clf()"""

        levelStrengths = []
        for unLev in uniqLevels:
            levelData = dataPoints[levels == unLev]
            lvlStren = levelData.shape[0] / dataPoints.shape[0]
            levelStrengths.append(lvlStren)

        levelFeatures = levelVals + levelAngles + levelStrengths

        return levelFeatures

    def findBestTrendLine(self, dataPoints, levels, render=False):

        dataScaler = MinMaxScaler()

        dataPoints = dataPoints.reshape((-1, 1))
        dataScaler.fit(dataPoints)
        dataPoints = dataScaler.transform(dataPoints)
        dataPoints = dataPoints.reshape((-1))

        #score start data points by rfm
        startTrendScores = self.rfmScorePoints(dataPoints, levels, inverseR=True, inverseM=True)
        #score end data points by rfm
        endTrendScores = self.rfmScorePoints(dataPoints, levels, inverseR=False, inverseM=False)
        #select 2 different levels
        #find there most relevant start and end trend points
        uniqLevels = np.sort(np.unique(levels))
        trendPoints = []
        trendPointsScores = []
        for i in range(uniqLevels.shape[0]):
            for j in range(uniqLevels.shape[0]):
                if i != j:
                    levelPoints_1 = dataPoints[ levels == uniqLevels[i] ]
                    levelScores_1 = startTrendScores[ levels == uniqLevels[i] ]
                    levelX_1 = np.arange(dataPoints.shape[0])[ levels == uniqLevels[i] ] / dataPoints.shape[0]
                    levelPoints_2 = dataPoints[ levels == uniqLevels[j] ]
                    levelScores_2 = endTrendScores[levels == uniqLevels[j]]
                    levelX_2 = np.arange(dataPoints.shape[0])[levels == uniqLevels[j]] / dataPoints.shape[0]
                    x_1, y_1, pointScore_1 = self.getBestLevelPoint(levelX_1, levelPoints_1, levelScores_1, startPoint=True)
                    x_2, y_2, pointScore_2 = self.getBestLevelPoint(levelX_2, levelPoints_2, levelScores_2, startPoint=False)
                    if x_1 < x_2:
                        trendPoints.append( [x_1, y_1, x_2, y_2] )
                        trendPointsScores.append( (pointScore_1 + pointScore_2) / 2 )

        #collect trend info like: avg rfm score of start/end points, len of trend, angle of trend
        #select best trend line (normalize )
        trendLens = []
        trendAngles = []
        for i in range(len(trendPoints)):
            x_1, y_1, x_2, y_2 = trendPoints[i]
            trendLens.append( self.getTrendLength(x_1, y_1, x_2, y_2) )
            trendAngles.append( self.getTrendAngle(x_1, y_1, x_2, y_2) )

        trendLens = np.array(trendLens).reshape((-1, 1))
        trendLens = MinMaxScaler().fit_transform(trendLens)
        trendLens = trendLens.reshape((-1))
        trendAngles = np.array(trendAngles).reshape((-1, 1))
        trendAngles = MinMaxScaler().fit_transform(trendAngles)
        trendAngles = trendAngles.reshape((-1))

        trendScores = np.zeros(len(trendLens))
        for i in range(len(trendLens)):
            trendScores[i] = (trendLens[i] + trendAngles[i]) / 2

        bestTrendPoints = trendPoints[np.argmax(trendScores)]
        bestTrendPoints[0] *= dataPoints.shape[0]
        bestTrendPoints[2] *= dataPoints.shape[0]
        bestTrendPoints[1] = dataScaler.inverse_transform(np.array(bestTrendPoints[1]).reshape((-1, 1)))[0][0]
        bestTrendPoints[3] = dataScaler.inverse_transform(np.array(bestTrendPoints[3]).reshape((-1, 1)))[0][0]

        if render:
            dataPoints = dataPoints.reshape((-1, 1))
            dataPoints = dataScaler.inverse_transform(dataPoints)
            dataPoints = dataPoints.reshape((-1))
            plt.scatter(np.arange(dataPoints.shape[0]), dataPoints, c=levels, s=5)
            plt.plot([bestTrendPoints[0], bestTrendPoints[2]], [bestTrendPoints[1], bestTrendPoints[3]])
            for unLev in uniqLevels:
                levelVals = dataPoints[levels == unLev]
                levelVal = np.average(levelVals)
                # print(levelVals.shape)
                plt.axhline(y=levelVal, c="black", linestyle="-")
            plt.savefig("./images/{}.png".format(str(uuid.uuid4())), dpi=300)
            plt.clf()

        return bestTrendPoints

    def getTrendLength(self, x_1, y_1, x_2, y_2):
        trendLen = np.sqrt( np.square(x_2 - x_1) + np.square(y_2 - y_1) )
        return trendLen

    def getTrendAngle(self, x_1, y_1, x_2, y_2):

        x = x_2 - x_1
        y = y_2 - y_1

        cosX = x / np.sqrt( np.square(x) + np.square(y) )

        sinX = np.sqrt(1 - np.square(cosX))

        return sinX

    def getBestLevelPoint(self, xCoors, dataPoints, scores, startPoint):

        df = pd.DataFrame({"x":xCoors, "y": dataPoints, "scores": scores})
        df["s"] = pd.qcut(df["scores"], 4, labels=False, duplicates="drop") + 1
        #print(df.groupby("s")["scores"].agg(["mean", "count"]))

        pointScore = df["s"].max()
        bestPoints = df[ df["s"] == pointScore ]

        if startPoint:
            bestX = bestPoints["x"].min()

            #bestY = df["y"].min()
            bestX = df["x"].mean()
        else:
            bestX = bestPoints["x"].max()

            #bestY = df["y"].max()
            bestX = df["x"].mean()
        bestY = df["y"].mean()

        return bestX, bestY, pointScore

    def rfmScorePoints(self, dataPoints, levels, inverseR, inverseM):

        r = self.getR(dataPoints, inverse=inverseR)
        f = self.getF(dataPoints, levels)
        m = self.getM(dataPoints, inverse=inverseM)

        rfmScores = (r + f + m) / 3.0

        return rfmScores

    def getR(self, dataPoints, inverse=False):

        #use only 4 levels of rfm score
        nPoints = dataPoints.shape[0]
        h = nPoints // 4
        limits = []
        for i in range(4):
            limits.append( (i+1)*h )
        limits[3] = nPoints

        rScores = np.zeros(nPoints)
        for i in range(nPoints):
            for j in range(4):
                if i <= limits[j]:
                    if not inverse:
                        rScores[i] = j + 1
                    else:
                        rScores[i] = 4 - j
                    break

        return rScores

    def getF(self, dataPoints, levels):
        uniqLevels = np.sort(np.unique(levels))

        countDict = {}
        for unLev in uniqLevels:
            levelPoints = dataPoints[ levels == unLev ]
            countDict[unLev] = levelPoints.shape[0]

        orderedFreqs = list(sorted(countDict.values()))
        freqRatesDict = {}
        for i in range(len(uniqLevels)):
            freqRatesDict[orderedFreqs[i]] = i + 1

        fScoreDict = {}
        for unLev in uniqLevels:
            fScoreDict[unLev] = freqRatesDict[ countDict[unLev] ]

        fScores = np.zeros((levels.shape[0]))
        for i in range(fScores.shape[0]):
            fScores[i] = fScoreDict[levels[i]]

        return fScores

    def getM(self, dataPoints, inverse=False):

        df = pd.DataFrame({"data": dataPoints})
        df["m"] = pd.qcut(df["data"], 4, labels=False, duplicates="drop") + 1
        #print(df.groupby("m")["data"].agg(["mean", "count"]))

        mScores = df["m"].values
        if inverse:
            invDict = {}
            for i in range(4):
                invDict[i+1] = 4 - i

            for i in range(mScores.shape[0]):
                mScores[i] = invDict[mScores[i]]

        return mScores

    def plotLevels(self, avgVals, levels):

        uniqLevels = np.sort(np.unique(levels))
        for unLev in uniqLevels:
            levelVals = avgVals[levels == unLev]
            levelVal = np.average(levelVals)
            #print(levelVals.shape)
            plt.axhline(y=levelVal, c="black", linestyle="-")

            # lowLim = np.min(levelVals)
            # plt.axhline(y=lowLim, c="red", linestyle="--", linewidth=0.5)

            # highLim = np.max(levelVals)
            # plt.axhline(y=highLim, c="green", linestyle="--", linewidth=0.5)

        plt.scatter([x for x in range(avgVals.shape[0])], avgVals, c=levels, s=5)
        plt.show()


