
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from datetime import datetime
from avera_core.CloserFeatureGenerator import CloserFeatureGenerator
from avera.utils.save_load import *
from tqdm import tqdm

startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
dataManager = SymbolDataManager("../data/raw/")
df = dataManager.getData(symbol, timeframe)

df = df.tail(50000)
print(df.shape)
print(df.head(2))
print(df.tail(2))

df.set_index("datetime", drop=True, inplace=True)

featGen = CloserFeatureGenerator(featureList=["open", "close"],
                                 nFeatRows=1,
                                 nPoints=110,
                                 nLevels=5,
                                 flatStack=True,
                                 fitOnStep=True)

featGen = featGen.globalFit(df)
startTime = datetime.now()
#for i in tqdm(range(800)):
#    featImage = featGen.getManyRowsFeat(df.index.values[230 + i], df, None)
#    #featImage = featGen.getOneRowFeat(df.index.values[165 + i], df, None)
#endTime = datetime.now()
#print(str(endTime - startTime))
save("./closerFeatGen.pkl", featGen, verbose=True)
featGen = load("./closerFeatGen.pkl")
print("done")