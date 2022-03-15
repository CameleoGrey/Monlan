

from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.FeatGen_PriceEMA import FeatGen_PriceEMA
from classes.delta_bender.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from sklearn.preprocessing import MinMaxScaler
from classes.delta_bender.PlotRender import PlotRender


class FeatGenForRL():
    def __init__(self , featureList, nPoints = 256, window_size = 64, ema_period_volume = 14, ema_period_price = 3):
        self.nPoints = nPoints
        self.featureList = featureList

        self.window_size = window_size
        self.ema_period_volume = ema_period_volume
        self.ema_period_price = ema_period_price
        self.fg_sw = FeatGen_ScaledWindow(["open", "high", "low", "close", "cdv"], nPoints=self.window_size, nDiffs=0, flatStack=False)
        self.fg_cdv = FeatGen_CDV()
        self.fg_pe = FeatGen_PriceEMA()

        self.featureShape = (self.window_size, 5 )


        pass

    def globalFit(self, df):
        pass

    def getFeatByDatetime(self, datetimeStr, historyData, expandDims=True):
        x = self.getManyPointsFeat(datetimeStr, historyData)
        return x

    def getManyPointsFeat(self, ind, historyData ):
        df = historyData.copy()
        df = df[["open", "close", "low", "high", "tick_volume"]]

        #obs = self.get_price_ema_obs_(ind, df)

        obs = self.get_heiken_cdv_obs_(ind, df)

        return obs

    def get_price_ema_obs_(self, ind, df):
        df = df.loc[:ind]
        df = df.tail(self.nPoints).copy()

        df = self.fg_cdv.transform(df, self.ema_period_volume)
        df = df.iloc[self.ema_period_volume - 1:]

        df = self.fg_pe.transform(df, self.ema_period_price, verbose=False)
        df = df.iloc[self.ema_period_price - 1:]

        del df["tick_volume"]

        ind = df.index.values[-1]
        obs = self.fg_sw.get_window(ind, df)

        return obs

    def get_heiken_cdv_obs_(self, ind, df):
        df = df.loc[:ind]
        df = df.tail(self.nPoints).copy()
        df = FeatGen_HeikenAshi().transform(df, verbose=False)

        df = self.fg_cdv.transform(df, self.ema_period_volume)
        df = df.iloc[self.ema_period_volume - 1:]
        df = df.tail(self.window_size)
        del df["tick_volume"]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        for feat in ["open", "high", "low", "close"]:
            tmp = df[feat].values.reshape((-1, 1))
            scaler.partial_fit(tmp)
        for feat in ["open", "high", "low", "close"]:
            tmp = df[feat].values.reshape((-1, 1))
            tmp = scaler.transform(tmp)
            tmp = tmp.reshape((-1,))
            df[feat] = tmp

        for feat in ["cdv"]:
            tmp = df[feat].values.reshape((-1, 1))
            tmp = MinMaxScaler(feature_range=(-1, 1)).fit_transform(tmp)
            tmp = tmp.reshape((-1,))
            df[feat] = tmp

        #PlotRender().plot_price_cdv(df)

        obs = df.values
        obs = obs.reshape((1,) + obs.shape + (1,))

        return obs

    def getMinDate(self, df):
        minDate = df.index.values[self.nPoints]
        return minDate