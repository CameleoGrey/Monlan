
from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from matplotlib import pyplot as plt

df = SymbolDataManager().getData("EURUSD_i", "H1")
featureScaler = FeatureScaler()
df = featureScaler.extractFeature(df, featList=["open","high","low","close"])
print(df)

print( df["open"].mean() )
print( df["open"].std() )

y = df["open"].values
x = [x for x in range(y.shape[0])]
plt.plot( x, y )

y = df["close"].values
x = [x for x in range(y.shape[0])]
plt.plot( x, y )

plt.show()