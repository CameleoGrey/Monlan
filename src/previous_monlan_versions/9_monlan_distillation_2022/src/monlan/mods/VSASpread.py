
import matplotlib.pyplot as plt
import pandas as pd

class VSASpread():
    def __init__(self):
        pass

    def mod_history(self, df):
        mod_df = df.copy()
        close_col = mod_df["close"].values
        high_col = mod_df["high"].values
        low_col = mod_df["low"].values
        for i in range(mod_df.shape[0]):
            #currentRow = mod_df.iloc[i].copy()
            #currentRow["close"] = currentRow["high"] - currentRow["low"]
            #mod_df.iloc[i] = currentRow
            close_col[i] = high_col[i] - low_col[i]
            #if i % (mod_df.shape[0] // 20) == 0:
            #    print( "VSA mod: {:.2%}".format(i / mod_df.shape[0]) )
        mod_df["close"] = close_col
        mod_df.rename(columns={"close": "vsa_spread"}, inplace=True)
        #mod_df.drop( ["open", "high", "low", "real_volume", "spread", "tick_volume", "datetime"], axis=1, inplace=True )

        raw_df = df.copy()
        #for col in mod_df.columns:
        #    raw_df[col] = mod_df[col]
        raw_df["vsa_spread"] = mod_df["vsa_spread"]

        return raw_df

    def check_quality(self, df):

        fig, ax = plt.subplots(nrows=2, ncols=1)

        en_open = df["vsa_spread"].values
        x = [x for x in range(df.shape[0])]
        ax[0].plot(x, en_open)

        rawOpen = df["close"].values
        ax[1].plot(x, rawOpen)

        plt.show()
        pass