
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
from copy import deepcopy
import time
import pytz

class SymbolDataUpdater():
    def __init__(self, rawDataDir=None):
        if rawDataDir is None:
            self.rawDataDir = "../data/raw/"
        else:
            self.rawDataDir = rawDataDir
        pass

    def fullUpdate(self, terminal, symbol, timeFrame, startDate="2020-01-01 00:00:00"):

        timezone = pytz.timezone("Etc/UTC")
        startDate = parser.parse(startDate).astimezone(timezone) + timedelta(hours=3)
        endDate = datetime.now(timezone) + timedelta(hours=3)

        dataUpdate = terminal.copy_rates_range(symbol=symbol, timeframe=timeFrame,
            date_from=startDate, date_to=endDate)

        dataUpdate = pd.DataFrame(dataUpdate)
        dataUpdate['time'] = pd.to_datetime(dataUpdate['time'], unit='s')
        columns = list(dataUpdate.columns)
        columns[0] = "datetime"
        dataUpdate.columns = columns

        dataUpdate.to_csv(self.rawDataDir + symbol + "_" + timeFrame + ".csv", sep="\t", index=False)

        return True

    def partialUpdate(self, terminal, symbol, timeFrame, execution="correct"):

        existedData = pd.read_csv(self.rawDataDir + symbol + "_" + timeFrame + ".csv", sep="\t")
        lastRows = existedData.tail(2)
        prevLastDate = parser.parse( lastRows.iloc[0]["datetime"] ) #+ timedelta(hours=3)
        lastDate = parser.parse(lastRows.iloc[1]["datetime"]) #+ timedelta(hours=3)
        td = lastDate - prevLastDate
        updateStartDate = lastDate - td

        timezone = pytz.timezone("Etc/UTC")
        endDate = datetime.now(timezone) + timedelta(hours=3)

        if execution == "correct":
            tmp = deepcopy(updateStartDate)
            tmp = tmp.astimezone(timezone) + timedelta(hours=3)
            while int(tmp.timestamp()) < int(endDate.timestamp()):
                tmp += td
            endDate = tmp - timedelta(seconds=10)
            updateStartDate = updateStartDate - 1000 * td

            #wait for the correct date to update
            currentTime = datetime.now(timezone) + timedelta(hours=3)
            while int( currentTime.timestamp() ) < int(endDate.timestamp()):
                time.sleep(1)
                currentTime = datetime.now(timezone) + timedelta(hours=3)
            print("time of update: {}".format(currentTime))


        dataUpdate = terminal.copy_rates_range(symbol=symbol, timeframe=timeFrame,
            date_from=updateStartDate, date_to=endDate)

        dataUpdate = pd.DataFrame(dataUpdate)
        if len(dataUpdate) == 0:
            return True
        dataUpdate['time'] = pd.to_datetime(dataUpdate['time'], unit='s')
        columns = list(dataUpdate.columns)
        columns[0] = "datetime"
        dataUpdate.columns = columns

        existedData.datetime = pd.to_datetime( existedData.datetime, format = "%Y-%m-%d %H:%M:%S" )
        existedData = existedData[ existedData.datetime < dataUpdate.datetime[0] ]

        existedData = existedData.append( dataUpdate )
        existedData.to_csv(self.rawDataDir + symbol + "_" + timeFrame + ".csv", sep="\t", index=False)
        return True