
import numpy as np
from tqdm import tqdm

class FeatGen_CDV():
    def __init__(self):
        pass

    def fit(self, df):
        return self

    def transform(self, df, period=14, verbose=False, add_raw_delta=False):

        df = df.copy()

        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["tick_volume"].values

        delta_vals = []
        if verbose:
            cycle_range = tqdm(range(open.shape[0]), desc="delta", colour="green")
        else:
            cycle_range = range(open.shape[0])
        for i in cycle_range:
            delta = self.get_delta_( open[i], high[i], low[i], close[i], volume[i] )
            delta_vals.append( delta )

        if add_raw_delta:
            df["delta"] = delta_vals

        nan_vals = []

        for i in range( period - 1 ):
            nan_vals.append( 0 )
        cum_delta = self.ema_( delta_vals, period )

        #for i in range( 3*period - 3 ):
        #    nan_vals.append( 0 )
        #cum_delta = self.tema_(delta_vals, period)

        cum_delta = np.hstack( [nan_vals, cum_delta] )
        df["cdv"] = cum_delta

        return df

    def get_delta_(self, open, high, low, close, volume):

        upper_wick = high - close if close > open else high - open
        lower_wick = open - low if close > open else close - low
        spread = high - low
        body_length = spread - (upper_wick + lower_wick)

        if spread == 0:
            percent_upper_wick = 0.0
            percent_lower_wick = 0.0
            percent_body_length = 0.0
        else:
            percent_upper_wick = upper_wick / spread
            percent_lower_wick = lower_wick / spread
            percent_body_length = body_length / spread

        buying_volume = 0.0
        selling_volume = 0.0
        if close > open:
            buying_volume = (percent_body_length + (percent_upper_wick + percent_lower_wick) / 2) * volume
            selling_volume = ((percent_upper_wick + percent_lower_wick) / 2) * volume
        elif close < open:
            buying_volume = ((percent_upper_wick + percent_lower_wick) / 2) * volume
            selling_volume = (percent_body_length + (percent_upper_wick + percent_lower_wick) / 2) * volume

        delta = buying_volume - selling_volume
        return delta

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