
import matplotlib.pyplot as plt
import pandas as pd

class EnergyMod():
    def __init__(self):
        pass

    def mod_history(self, df, feat_list):

        #feat_list.append("tick_volume")
        raw_df = df.copy()
        n_diffs = 0
        for i in range(n_diffs):
            for feat in feat_list:
                not_shifted = raw_df[feat]
                shifted_data = raw_df[feat].shift(periods=1)
                raw_df[feat] = not_shifted - shifted_data
            iter = next(raw_df.iterrows())
            raw_df = raw_df.drop(iter[0])
        #feat_list.remove("tick_volume")

        #raw_df.reset_index(drop=True, inplace=True)
        mod_df = raw_df.copy()
        #mod_df.reset_index(drop=True, inplace=True)
        n_steps = 0
        en_feat_dict = {}
        for feat in feat_list:
            en_feat_dict[feat] = []
        for current_row in mod_df.iterrows():
            #current_row = mod_df.iloc[i].copy()
            for feat in feat_list:
                sign = 1
                if current_row[1][feat] < 0:
                    sign = -1
                en_feat_dict[feat].append( sign * ((current_row[1][feat] ** 2) / 2) * abs(current_row[1]["tick_volume"]) )
            n_steps += 1
            #if n_steps % (mod_df.shape[0] // 20) == 0:
            #    print( "Energy mod: {:.2%}".format(n_steps / mod_df.shape[0]) )
        for feat in feat_list:
            mod_df[feat] = en_feat_dict[feat]
        #mod_df.reset_index(drop=True, inplace=True)
        col_dict = {}
        for feat in feat_list:
            col_dict[feat] = "en" + feat
        mod_df.rename(columns=col_dict, inplace=True)
        col_to_remove = ["open", "high", "low", "close"]
        for feat in feat_list:
            col_to_remove.remove(feat)
        mod_df.drop(col_to_remove, axis=1, inplace=True)
        #mod_df.drop( ["real_volume", "spread", "tick_volume", "datetime"], axis=1, inplace=True )

        raw_df = df.copy()
        #raw_df.reset_index(drop=True, inplace=True)
        raw_df = raw_df.drop(raw_df.index.values[0])
        #raw_df.reset_index(drop=True, inplace=True)

        for feat in feat_list:
            raw_df[col_dict[feat]] = mod_df[col_dict[feat]]

        return raw_df

    def check_quality(self, df):

        fig, ax = plt.subplots(nrows=2, ncols=1)

        en_open = df["enopen"].values
        x = [x for x in range(df.shape[0])]
        ax[0].plot( x, en_open )

        raw_open = df["open"].values
        ax[1].plot(x, raw_open)

        plt.show()
        pass