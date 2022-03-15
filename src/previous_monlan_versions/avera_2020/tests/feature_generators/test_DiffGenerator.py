
from avera.feature_generators.DiffGenerator import DiffGenerator
from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from matplotlib import pyplot as plt
from dateutil import parser

featureList = ["open","close"]
df = SymbolDataManager().getData("TEST", "H1", normalizeDateTime=True, normalizeNames=True)
df.set_index("datetime", drop=True, inplace=True)
featureScaler = DiffGenerator(featureList=featureList, nDiffs=1, nPoints=4, flatStack=True)
featureScaler = featureScaler.globalFit(df)

print( df.tail(1) )
lastDt = list(df.tail(1).index)[0]
print( lastDt )
print( df.loc[lastDt] )
print( featureScaler.getFeatByDatetime( lastDt, df) )

featureScaler = FeatureScaler()
df = featureScaler.extractFeature(df, featureList)
print( df.loc[lastDt] )