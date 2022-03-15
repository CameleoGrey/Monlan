
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta

nMeasures = 1000
#fill date and time columns
dateList = []
timeList = []
startDate = datetime(year=2016, month=1, day=4, hour=1, minute=0, second=0)
td = timedelta(hours=1)
for i in range(nMeasures):
    dateList.append( str(startDate.date()) )
    timeList.append( str(startDate.time()) )
    startDate += td

#fill open, close, high, low
openList = []
closeList = []
highList = []
lowList = []
startPoint = 0.0
deltaPoint = 20 * 2 * np.pi / nMeasures
for i in range(nMeasures // 2):
    openList.append( round(i + 1000 * np.sin(startPoint), 6) )
    closeList.append( round( 1.20*(i + 1000 * np.sin(startPoint) ), 6) )
    highList.append( round(i + 1000 * np.sin(startPoint) + 2.0, 6) )
    lowList.append( round(i + 1000 * np.sin(startPoint) - 2.0, 6) )
    startPoint += deltaPoint
for i in range(nMeasures // 2, 0, -1):
    openList.append( round(i + 1000 * np.sin(startPoint), 6) )
    closeList.append( round(0.8*(i + 1000 * np.sin(startPoint) ), 6) )
    highList.append( round(i + 1000 * np.sin(startPoint) + 2.0, 6) )
    lowList.append( round(i + 1000 * np.sin(startPoint) - 2.0, 6) )
    startPoint += deltaPoint

#fill tickvol
tickList = []
meanTickVol = 200
for i in range(nMeasures):
    tickList.append(int(meanTickVol + 0.5 * np.random.standard_normal(1)[0] * meanTickVol))

#fill vol, spread
volList = []
spreadList = []
meanVol = 10
meanSpread = 14
for i in range(nMeasures):
    volList.append(int(meanVol + 0.5 * np.random.standard_normal(1)[0] * meanVol))
    spreadList.append(int(meanSpread + 0.5 * np.random.standard_normal(1)[0] * meanSpread))

columns = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<TICKVOL>", "<VOL>", "<SPREAD>"]
testDataFrame = pd.DataFrame( data={"<DATE>": dateList,
                                    "<TIME>": timeList,
                                    "<OPEN>": openList,
                                    "<HIGH>": highList,
                                    "<LOW>": lowList,
                                    "<CLOSE>": closeList,
                                    "<TICK_VOLUME>": tickList,
                                    "<REAL_VOLUME>": volList,
                                    "<SPREAD>": spreadList})
print(testDataFrame)

#save test data
testDataFrame.to_csv( "../../data/raw/" + "TEST_H1.csv", sep="\t", index=False )
print("test data created: {}".format("../data/raw/" + "TEST_H1.csv"))