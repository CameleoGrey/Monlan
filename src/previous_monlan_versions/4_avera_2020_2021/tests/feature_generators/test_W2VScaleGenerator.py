
from avera.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from matplotlib import pyplot as plt
from dateutil import parser

featureList = ["open","close"]
df = SymbolDataManager().getData("TEST", "H1", normalizeDateTime=True, normalizeNames=True)
df.set_index("datetime", drop=True, inplace=True)
featureGenerator = W2VScaleGenerator(featureList=featureList, nPoints = 10, flatStack = False, fitOnStep = False,
                 nIntervals = 1000, w2vSize=50, window=20, iter=20, min_count=0, sample=0.0, sg=0)
featureGenerator = featureGenerator.globalFit(df)
featureGenerator.saveGenerator("./w2vGen.pkl")
featureGenerator = W2VScaleGenerator(featureList=featureList).loadGenerator("./w2vGen.pkl")
featureGenerator.checkReconstructionQuality(df)

print( df.tail(1) )
lastDt = list(df.tail(1).index)[0]
print( lastDt )
print( df.loc[lastDt] )
print( featureGenerator.getFeatByDatetime( lastDt, df) )
