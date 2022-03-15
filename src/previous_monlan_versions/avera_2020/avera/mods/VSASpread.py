
import matplotlib.pyplot as plt
import pandas as pd

class VSASpread():
    def __init__(self):
        pass

    def modHistory(self, df):
        modDf = df.copy()
        closeCol = modDf["close"].values
        highCol = modDf["high"].values
        lowCol = modDf["low"].values
        for i in range(modDf.shape[0]):
            #currentRow = modDf.iloc[i].copy()
            #currentRow["close"] = currentRow["high"] - currentRow["low"]
            #modDf.iloc[i] = currentRow
            closeCol[i] = highCol[i] - lowCol[i]
            #if i % (modDf.shape[0] // 20) == 0:
            #    print( "VSA mod: {:.2%}".format(i / modDf.shape[0]) )
        modDf["close"] = closeCol
        modDf.rename(columns={"close": "vsa_spread"}, inplace=True)
        #modDf.drop( ["open", "high", "low", "real_volume", "spread", "tick_volume", "datetime"], axis=1, inplace=True )

        rawDf = df.copy()
        #for col in modDf.columns:
        #    rawDf[col] = modDf[col]
        rawDf["vsa_spread"] = modDf["vsa_spread"]

        return rawDf

    def checkQuality(self, df):

        fig, ax = plt.subplots(nrows=2, ncols=1)

        enOpen = df["vsa_spread"].values
        x = [x for x in range(df.shape[0])]
        ax[0].plot(x, enOpen)

        rawOpen = df["close"].values
        ax[1].plot(x, rawOpen)

        plt.show()
        pass