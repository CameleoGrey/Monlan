
from avera.feature_generators.FeatureDiff import FeatureDiff
from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from matplotlib import pyplot as plt

df = SymbolDataManager().getData("EURUSD_i", "H1")
df = FeatureDiff().extractFeature(df, featList=["open"], nDiffs=1)
df = FeatureScaler().extractFeature(df, featList=["open"])
print(df)

print( df["open"].mean() )
print( df["open"].std() )

y = df["open"].values
x = [x for x in range(y.shape[0])]
plt.plot( x, y )
plt.show()