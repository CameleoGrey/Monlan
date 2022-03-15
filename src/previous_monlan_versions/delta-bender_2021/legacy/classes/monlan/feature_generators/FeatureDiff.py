
from sklearn.preprocessing import StandardScaler
import numpy as np

class FeatureDiff:
    def __init__(self):
        pass

    def extractFeature(self, df, featList=[], nDiffs = 1):
        dfCols = list(df.columns)
        for feat in featList:
            if feat not in dfCols:
                raise ValueError("df doesn't contain column: \"{}\" ".format(feat))

        if len( featList ) == 0:
            featList = list( df.columns )
            featList.remove("datetime")
        for i in range(nDiffs):
            for feat in featList:
                notShifted = df[feat]
                shiftedData = df[feat].shift(periods=1)
                df[feat] = notShifted - shiftedData
            iter = next(df.iterrows())
            df = df.drop(iter[0])

        return df