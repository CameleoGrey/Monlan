
import matplotlib.pyplot as plt
import pandas as pd

class HeikenAshiMod():
    def __init__(self):
        pass

    def mod_history(self, df):
        raw_df = df.copy()
        #raw_df.reset_index(drop=True, inplace=True)
        mod_df = df.copy()
        #mod_df.reset_index(drop=True, inplace=True)
        raw_close_col = raw_df["close"].copy().values
        raw_open_col = raw_df["open"].copy().values
        mod_close_col = mod_df["close"].copy().values
        mod_open_col = mod_df["open"].copy().values
        mod_high_col = mod_df["high"].copy().values
        mod_low_col = mod_df["low"].copy().values
        for i in range(1, mod_df.shape[0]):
            mod_close_col[i] = (mod_open_col[i] + mod_close_col[i] + mod_low_col[i] + mod_high_col[i]) / 4
            mod_open_col[i] = (raw_open_col[i-1] + raw_close_col[i-1]) / 2
            #if i % (mod_df.shape[0] // 20) == 0:
            #    print( "Heiken Ashi mod: {:.2%}".format(i / mod_df.shape[0]) )
        mod_df["close"] = mod_close_col
        mod_df["open"] = mod_open_col
        mod_df = mod_df.drop(mod_df.index.values[0])
        #mod_df.reset_index(drop=True, inplace=True)
        mod_df.rename(columns={"open": "hkopen", "close": "hkclose"}, inplace=True)
        #mod_df.drop( ["high", "low", "real_volume", "spread", "tick_volume", "datetime"], axis=1, inplace=True )

        raw_df = df.copy()
        #raw_df.reset_index(drop=True, inplace=True)
        raw_df = raw_df.drop(raw_df.index.values[0])
        #raw_df.reset_index(drop=True, inplace=True)

        #for col in mod_df.columns:
        #    raw_df[col] = mod_df[col]
        raw_df["hkopen"] = mod_df["hkopen"]
        raw_df["hkclose"] = mod_df["hkclose"]

        return raw_df

    def check_quality(self, df):

        hk_open = df["hkopen"].values
        hk_close = df["hkclose"].values
        hk_high = df["high"].values
        hk_low = df["low"].values
        for i in range(df.shape[0]):
            open = hk_open[i]
            close = hk_close[i]
            if close > open:
                color = "green"
                plt.plot( [i, i], [open, close], c=color, linewidth=2.0 )
                up_line = [hk_close[i], hk_high[i]]
                down_line = [hk_open[i], hk_low[i]]
                plt.plot([i, i], up_line, c="black", linewidth=0.5)
                plt.plot([i, i], down_line, c="black", linewidth=0.5)
            else:
                color = "red"
                plt.plot([i, i], [close, open], c=color, linewidth=2.0)
                up_line = [hk_open[i], hk_high[i]]
                down_line = [hk_close[i], hk_low[i]]
                plt.plot([i, i], up_line, c="black", linewidth=0.5)
                plt.plot([i, i], down_line, c="black", linewidth=0.5)
            if i % (df.shape[0] // 20) == 0:
                print( "Processing: {:.2%}".format(i / df.shape[0]) )
        plt.show()

        pass