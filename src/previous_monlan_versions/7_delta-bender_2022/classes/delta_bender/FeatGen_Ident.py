
class FeatGen_Ident():
    def __init__(self , featureList, nPoints = 200):
        self.nPoints = nPoints
        self.featureList = featureList
        self.featureShape = (-1,)
        pass

    def globalFit(self, df):
        pass

    def getFeatByDatetime(self, datetimeStr, historyData, expandDims=True):
        x = self.getManyPointsFeat(datetimeStr, historyData)
        return x

    def getManyPointsFeat(self, ind, historyData ):
        df = historyData.copy()

        df = df.loc[:ind]
        df = df.tail(self.nPoints).copy()


        return df

    def getMinDate(self, df):
        minDate = df.index.values[self.nPoints]
        return minDate