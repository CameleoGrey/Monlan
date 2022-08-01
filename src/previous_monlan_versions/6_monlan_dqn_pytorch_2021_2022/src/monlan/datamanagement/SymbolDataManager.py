
import pandas as pd
from datetime import datetime
from dateutil import parser

class SymbolDataManager:
    def __init__(self, raw_data_dir=None):
        if raw_data_dir is None:
            self.raw_data_dir = "../data/raw/"
        else:
            self.raw_data_dir = raw_data_dir
        pass

    def get_data(self, symbol, time_frame, normalize_names = False, normalize_date_time = False):
        readed_data = pd.read_csv( self.raw_data_dir + symbol + "_" + time_frame + ".csv", sep="\t")
        if normalize_names: readed_data = self.normalize_column_names(readed_data)
        if normalize_date_time: readed_data = self.normalize_date_timeColumns(readed_data)
        return readed_data

    def normalize_column_names(self, df):
        columns = df.columns.values
        for i in range( len(columns) ):
            columns[i] = columns[i].lower()
            columns[i] = columns[i].replace("<", "")
            columns[i] = columns[i].replace(">", "")
        df.columns = columns
        return df

    def normalize_date_timeColumns(self, df):
        datetime_column = pd.to_datetime( df["date"] + " " + df["time"] )
        del df["date"]
        del df["time"]
        df["datetime"] = datetime_column
        columns_names = list(df.columns.values)
        for i in range( len(columns_names) ):
            tmp = columns_names[i]
            columns_names[i] = columns_names[len(columns_names) - 1]
            columns_names[len(columns_names) - 1] = tmp
        df = df[columns_names]
        return df

