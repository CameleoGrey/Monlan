
from tqdm import tqdm

class FeatGen_HeikenAshi():
    def __init__(self):
        pass

    def fit(self, df):
        return self

    def transform(self, df, verbose=False):
        raw_df = df.copy()
        mod_df = df.copy()
        raw_close_col = raw_df["close"].copy().values
        raw_open_col = raw_df["open"].copy().values
        mod_close_col = mod_df["close"].copy().values
        mod_open_col = mod_df["open"].copy().values
        mod_high_col = mod_df["high"].copy().values
        mod_low_col = mod_df["low"].copy().values

        if verbose:
            cycle_range = tqdm( range(1, mod_df.shape[0]), desc="Heiken ashi", colour="green" )
        else:
            cycle_range = range(1, mod_df.shape[0])
        for i in cycle_range:
            mod_close_col[i] = (mod_open_col[i] + mod_close_col[i] + mod_low_col[i] + mod_high_col[i]) / 4
            mod_open_col[i] = (raw_open_col[i-1] + raw_close_col[i-1]) / 2
        mod_df["close"] = mod_close_col
        mod_df["open"] = mod_open_col
        mod_df = mod_df.drop(mod_df.index.values[0])

        return mod_df