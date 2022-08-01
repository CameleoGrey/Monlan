
import pandas as pd
from datetime import datetime
from dateutil import parser

class SymbolDataManager:
    def __init__(self, rawDataDir=None):
        if rawDataDir is None:
            self.rawDataDir = "../data/raw/"
        else:
            self.rawDataDir = rawDataDir
        pass

    def getData(self, symbol, timeFrame, normalizeNames = False, normalizeDateTime = False):
        readedData = pd.read_csv( self.rawDataDir + symbol + "_" + timeFrame + ".csv", sep="\t")
        if normalizeNames: readedData = self.normalizeColumnNames(readedData)
        if normalizeDateTime: readedData = self.normalizeDateTimeColumns(readedData)
        return readedData

    def normalizeColumnNames(self, df):
        columns = df.columns.values
        for i in range( len(columns) ):
            columns[i] = columns[i].lower()
            columns[i] = columns[i].replace("<", "")
            columns[i] = columns[i].replace(">", "")
        df.columns = columns
        return df

    def normalizeDateTimeColumns(self, df):
        datetimeColumn = pd.to_datetime( df["date"] + " " + df["time"] )
        del df["date"]
        del df["time"]
        df["datetime"] = datetimeColumn
        columnsNames = list(df.columns.values)
        for i in range( len(columnsNames) ):
            tmp = columnsNames[i]
            columnsNames[i] = columnsNames[len(columnsNames) - 1]
            columnsNames[len(columnsNames) - 1] = tmp
        df = df[columnsNames]
        return df

