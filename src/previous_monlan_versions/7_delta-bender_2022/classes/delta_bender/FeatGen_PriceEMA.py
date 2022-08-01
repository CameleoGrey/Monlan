
import numpy as np
from tqdm import tqdm

class FeatGen_PriceEMA():
    def __init__(self):
        pass

    def fit(self, df):
        return self

    def transform(self, df, period=3, verbose=True):

        df = df.copy()

        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        nan_vals = []
        for i in range( period - 1 ):
            nan_vals.append( 0 )

        ema_open = self.ema_( open, period )
        ema_high = self.ema_(high, period)
        ema_low = self.ema_(low, period)
        ema_close = self.ema_(close, period)

        ema_open = np.hstack( [nan_vals, ema_open] )
        ema_high = np.hstack( [nan_vals, ema_high] )
        ema_low = np.hstack( [nan_vals, ema_low] )
        ema_close = np.hstack( [nan_vals, ema_close] )

        df["open"] = ema_open
        df["high"] = ema_high
        df["low"] = ema_low
        df["close"] = ema_close

        return df

    def tema_(self, s, n):
        one_ema = self.ema_(s, n)
        double_ema = self.ema_( one_ema, n)
        triple_ema = self.ema_( double_ema, n)

        tema = 3 * one_ema[2*n-2:] - 3 * double_ema[n-1:] - triple_ema
        return tema

    def ema_(self, s, n):
        """
        returns an n period exponential moving average for
        the time series s

        s is a list ordered from oldest (index 0) to most
        recent (index -1)
        n is an integer

        returns a numeric array of the exponential
        moving average
        """
        ema = []
        j = 1

        # get n sma first and calculate the next n period ema
        sma = sum(s[:n]) / n
        multiplier = 2 / float(1 + n)
        ema.append(sma)

        # EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
        ema.append(((s[n] - sma) * multiplier) + sma)

        # now calculate the rest of the values
        for i in s[n + 1:]:
            tmp = ((i - ema[j]) * multiplier) + ema[j]
            j = j + 1
            ema.append(tmp)

        ema = np.array( ema )

        return ema