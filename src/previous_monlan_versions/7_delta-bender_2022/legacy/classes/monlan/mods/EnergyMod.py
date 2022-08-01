
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class EnergyMod():
    def __init__(self):
        pass

    def modHistory(self, df, featList):

        #featList.append("tick_volume")
        rawDf = df.copy()
        nDiffs = 0
        for i in range(nDiffs):
            for feat in featList:
                notShifted = rawDf[feat]
                shiftedData = rawDf[feat].shift(periods=1)
                rawDf[feat] = notShifted - shiftedData
            iter = next(rawDf.iterrows())
            rawDf = rawDf.drop(iter[0])
        #featList.remove("tick_volume")

        #rawDf.reset_index(drop=True, inplace=True)
        modDf = rawDf.copy()
        #modDf.reset_index(drop=True, inplace=True)
        enFeatDict = {}
        for feat in featList:
            enFeatDict[feat] = []
        for currentRow in tqdm(modDf.iterrows(), desc="Energy mod"):
            #currentRow = modDf.iloc[i].copy()
            for feat in featList:
                sign = 1
                if currentRow[1][feat] < 0:
                    sign = -1
                enFeatDict[feat].append( sign * ((currentRow[1][feat] ** 2) / 2) * abs(currentRow[1]["tick_volume"]) )
        for feat in featList:
            modDf[feat] = enFeatDict[feat]
        #modDf.reset_index(drop=True, inplace=True)
        colDict = {}
        for feat in featList:
            colDict[feat] = "en" + feat
        modDf.rename(columns=colDict, inplace=True)
        colToRemove = ["open", "high", "low", "close"]
        for feat in featList:
            colToRemove.remove(feat)
        modDf.drop(colToRemove, axis=1, inplace=True)
        #modDf.drop( ["real_volume", "spread", "tick_volume", "datetime"], axis=1, inplace=True )

        rawDf = df.copy()
        #rawDf.reset_index(drop=True, inplace=True)
        rawDf = rawDf.drop(rawDf.index.values[0])
        #rawDf.reset_index(drop=True, inplace=True)

        for feat in featList:
            rawDf[colDict[feat]] = modDf[colDict[feat]]

        return rawDf

    def checkQuality(self, df):

        fig, ax = plt.subplots(nrows=2, ncols=1)

        enOpen = df["enopen"].values
        x = [x for x in range(df.shape[0])]
        ax[0].plot_candles(x, enOpen)

        rawOpen = df["open"].values
        ax[1].plot_candles(x, rawOpen)

        plt.show()
        pass