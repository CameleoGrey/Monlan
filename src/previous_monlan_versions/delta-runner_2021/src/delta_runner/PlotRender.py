
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mpl_dates
from tqdm import tqdm

class PlotRender():
    def __init__(self):
        pass

    def plot_price_cdv(self, df):

        fig, ax = plt.subplots(2, 1, sharex=True)

        hkOpen = df["open"].values
        hkClose = df["close"].values
        hkHigh = df["high"].values
        hkLow = df["low"].values
        for i in tqdm(range(df.shape[0])):
            open = hkOpen[i]
            close = hkClose[i]
            if close > open:
                color = "green"
                ax[0].plot([i, i], [open, close], c=color, linewidth=2.0)
                upLine = [hkClose[i], hkHigh[i]]
                downLine = [hkOpen[i], hkLow[i]]
                ax[0].plot([i, i], upLine, c="black", linewidth=0.5)
                ax[0].plot([i, i], downLine, c="black", linewidth=0.5)
            else:
                color = "red"
                ax[0].plot([i, i], [close, open], c=color, linewidth=2.0)
                upLine = [hkOpen[i], hkHigh[i]]
                downLine = [hkClose[i], hkLow[i]]
                ax[0].plot([i, i], upLine, c="black", linewidth=0.5)
                ax[0].plot([i, i], downLine, c="black", linewidth=0.5)


        x = [x for x in range(len(df))]
        y = df["cdv"].values
        ax[1].bar(x, y, bottom=0.0)

        """def on_move(event):
            if event.inaxes == ax[0]:
                ax[1].view_init(elev=ax[0].elev, azim=ax[0].azim)
            elif event.inaxes == ax[1]:
                ax[0].view_init(elev=ax[1].elev, azim=ax[1].azim)
            else:
                return
            fig.canvas.draw_idle()

        c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)"""

        plt.show()

        pass

    def plot_assurance(self, df):

        fig, ax = plt.subplots(4, 1, sharex=True)

        hkOpen = df["open"].values
        hkClose = df["close"].values
        hkHigh = df["high"].values
        hkLow = df["low"].values
        for i in tqdm(range(df.shape[0])):
            open = hkOpen[i]
            close = hkClose[i]
            if close > open:
                color = "green"
                ax[0].plot([i, i], [open, close], c=color, linewidth=2.0)
                upLine = [hkClose[i], hkHigh[i]]
                downLine = [hkOpen[i], hkLow[i]]
                ax[0].plot([i, i], upLine, c="black", linewidth=0.5)
                ax[0].plot([i, i], downLine, c="black", linewidth=0.5)
            else:
                color = "red"
                ax[0].plot([i, i], [close, open], c=color, linewidth=2.0)
                upLine = [hkOpen[i], hkHigh[i]]
                downLine = [hkClose[i], hkLow[i]]
                ax[0].plot([i, i], upLine, c="black", linewidth=0.5)
                ax[0].plot([i, i], downLine, c="black", linewidth=0.5)

        x = [x for x in range(len(df))]
        y = df["assurance_bull"].values
        ax[1].bar(x, y, bottom=0.0, color="green")
        y = -df["assurance_bear"].values
        ax[1].bar(x, y, bottom=0.0, color="red")

        bull_assurance = df["assurance_bull"].values
        bear_assurance = df["assurance_bear"].values
        relative_assurance_bull = []
        relative_assurance_bear = []
        for i in range(len(bull_assurance)):
            bua = bull_assurance[i]
            bea = bear_assurance[i]
            if bua == 0 or bea == 0:
                relative_assurance_bull.append(0)
                relative_assurance_bear.append(0)
                continue
            ra = bua / bea if bua > bea else bea / bua
            #ra = ra - 1.0
            if bua > bea:
                relative_assurance_bull.append( ra )
                relative_assurance_bear.append( 0.0 )
            else:
                relative_assurance_bear.append( -ra )
                relative_assurance_bull.append(0.0)
        x = [x for x in range(len(df))]
        y = relative_assurance_bull
        ax[2].bar(x, y, bottom=0.0, color="green")
        y = relative_assurance_bear
        ax[2].bar(x, y, bottom=0.0, color="red")

        if "cdv" in df.columns:
            x = [x for x in range(len(df))]
            y = df["cdv"].values
            ax[3].bar(x, y, bottom=0.0)

        plt.show()

        pass

    def plot_candles(self, df):

        hkOpen = df["open"].values
        hkClose = df["close"].values
        hkHigh = df["high"].values
        hkLow = df["low"].values
        for i in tqdm(range(df.shape[0])):
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
        plt.show()

        pass

    def plot_cdv(self, df):

        x = [x for x in range(len(df))]
        y = df["cdv"].values
        plt.bar(x, y, bottom=0.0)
        plt.show()

        pass
