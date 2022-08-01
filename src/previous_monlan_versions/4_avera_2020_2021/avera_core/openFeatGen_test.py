
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from datetime import datetime
from tqdm import tqdm
from avera_core.TrendFinder import TrendFinder
from avera_core.OpenerFeatureGenerator import OpenerFeatureGenerator

startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
dataManager = SymbolDataManager("../data/raw/")
df = dataManager.getData(symbol, timeframe)

df = df.tail(150)
df.set_index("datetime", drop=True, inplace=True)

featGen = OpenerFeatureGenerator(featureList=["open", "close"],
                                 nFeatRows=1,
                                 nPoints=110,
                                 nLevels=4,
                                 flatStack=True,
                                 fitOnStep=True)

featGen = featGen.globalFit(df)
feats = featGen.getFeatByDatetime(df.index.values[109], df)
print(feats)
