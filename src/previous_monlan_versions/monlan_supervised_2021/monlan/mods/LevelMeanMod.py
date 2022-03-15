
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class HeikenAshiMod():
    def __init__(self):
        pass

    def modHistory(self, df):
        rawDf = df.copy()
        modDf = df.copy()
        rawCloseCol = rawDf["close"].copy().values
        rawOpenCol = rawDf["open"].copy().values
        modCloseCol = modDf["close"].copy().values
        modOpenCol = modDf["open"].copy().values
        modHighCol = modDf["high"].copy().values
        modLowCol = modDf["low"].copy().values
        for i in range(1, modDf.shape[0]):
            modCloseCol[i] = (modOpenCol[i] + modCloseCol[i] + modLowCol[i] + modHighCol[i]) / 4
            modOpenCol[i] = (rawOpenCol[i-1] + rawCloseCol[i-1]) / 2
            #if i % (modDf.shape[0] // 20) == 0:
            #    print( "Heiken Ashi mod: {:.2%}".format(i / modDf.shape[0]) )
        modDf["close"] = modCloseCol
        modDf["open"] = modOpenCol
        modDf = modDf.drop(modDf.index.values[0])
        #modDf.reset_index(drop=True, inplace=True)
        modDf.rename(columns={"open": "hkopen", "close": "hkclose"}, inplace=True)
        #modDf.drop( ["high", "low", "real_volume", "spread", "tick_volume", "datetime"], axis=1, inplace=True )

        rawDf = df.copy()
        #rawDf.reset_index(drop=True, inplace=True)
        rawDf = rawDf.drop(rawDf.index.values[0])
        #rawDf.reset_index(drop=True, inplace=True)

        #for col in modDf.columns:
        #    rawDf[col] = modDf[col]
        rawDf["hkopen"] = modDf["hkopen"]
        rawDf["hkclose"] = modDf["hkclose"]

        return rawDf

    def checkQuality(self, df):

        hkOpen = df["hkopen"].values
        hkClose = df["hkclose"].values
        hkHigh = df["high"].values
        hkLow = df["low"].values
        for i in range(df.shape[0]):
            open = hkOpen[i]
            close = hkClose[i]
            if close > open:
                color = "green"
                plt.plot( [i, i], [open, close], c=color, linewidth=2.0 )
                upLine = [hkClose[i], hkHigh[i]]
                downLine = [hkOpen[i], hkLow[i]]
                plt.plot([i, i], upLine, c="black", linewidth=0.5)
                plt.plot([i, i], downLine, c="black", linewidth=0.5)
            else:
                color = "red"
                plt.plot([i, i], [close, open], c=color, linewidth=2.0)
                upLine = [hkOpen[i], hkHigh[i]]
                downLine = [hkClose[i], hkLow[i]]
                plt.plot([i, i], upLine, c="black", linewidth=0.5)
                plt.plot([i, i], downLine, c="black", linewidth=0.5)
            if i % (df.shape[0] // 20) == 0:
                print( "Processing: {:.2%}".format(i / df.shape[0]) )
        plt.show()

        pass