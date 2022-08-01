
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from datetime import datetime
from tqdm import tqdm
from avera_core.TrendFinder import TrendFinder

startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
dataManager = SymbolDataManager("../data/raw/")
df = dataManager.getData(symbol, timeframe)

trendFinder = TrendFinder()
for i in tqdm(range(2000)):
    tmpDf = df.tail(7000).head(2500 + i)
    #dataPoints, levels = trendFinder.findLevels(tmpDf, nPoints=200)
    #x_1, y_1, x_2, y_2 = trendFinder.findBestTrendLine(dataPoints, levels, render=True)
    trendFinder.getLevelFeatures(tmpDf, 110)

