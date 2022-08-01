
from sklearn.preprocessing import StandardScaler
import numpy as np

class FeatureScaler:
    def __init__(self):
        pass

    def extractFeature(self, df, featList=[]):
        dfCols = list(df.columns)
        for feat in featList:
            if feat not in dfCols:
                raise ValueError("df doesn't contain column: \"{}\" ".format(feat))

        #if len( featList ) == 0:
        #    featList = list( df.columns )
        #    featList.remove("datetime")

        scaler = StandardScaler()
        for feat in featList:
            tmp = df[feat].values
            tmp = tmp.reshape(-1, 1)
            scaler = scaler.partial_fit(tmp)

        for feat in featList:
            data = df[feat].values
            data = data.reshape(-1, 1)
            data = scaler.transform(data)
            df[feat] = data

        return df