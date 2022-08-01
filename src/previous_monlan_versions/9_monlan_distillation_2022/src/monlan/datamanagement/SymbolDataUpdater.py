
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
from copy import deepcopy
import time
import pytz

class SymbolDataUpdater():
    def __init__(self, raw_data_dir=None):
        if raw_data_dir is None:
            self.raw_data_dir = "../data/raw/"
        else:
            self.raw_data_dir = raw_data_dir
        pass

    def full_update(self, terminal, symbol, time_frame, start_date="2020-01-01 00:00:00"):

        timezone = pytz.timezone("Etc/UTC")
        start_date = parser.parse(start_date).astimezone(timezone) + timedelta(hours=3)
        end_date = datetime.now(timezone) + timedelta(hours=3)

        data_update = terminal.copy_rates_range(symbol=symbol, timeframe=time_frame,
            date_from=start_date, date_to=end_date)

        data_update = pd.DataFrame(data_update)
        data_update['time'] = pd.to_datetime(data_update['time'], unit='s')
        columns = list(data_update.columns)
        columns[0] = "datetime"
        data_update.columns = columns

        data_update.to_csv(self.raw_data_dir + symbol + "_" + time_frame + ".csv", sep="\t", index=False)

        return True

    def partial_update(self, terminal, symbol, time_frame, execution="correct"):

        existed_data = pd.read_csv(self.raw_data_dir + symbol + "_" + time_frame + ".csv", sep="\t")
        last_rows = existed_data.tail(2)
        prev_last_date = parser.parse( last_rows.iloc[0]["datetime"] ) #+ timedelta(hours=3)
        last_date = parser.parse(last_rows.iloc[1]["datetime"]) #+ timedelta(hours=3)
        td = last_date - prev_last_date
        update_start_date = last_date - td

        timezone = pytz.timezone("Etc/UTC")
        end_date = datetime.now(timezone) + timedelta(hours=3)

        if execution == "correct":
            tmp = deepcopy(update_start_date)
            tmp = tmp.astimezone(timezone) + timedelta(hours=3)
            while int(tmp.timestamp()) < int(end_date.timestamp()):
                tmp += td
            end_date = tmp - timedelta(seconds=10)
            update_start_date = update_start_date - 1000 * td

            #wait for the correct date to update
            current_time = datetime.now(timezone) + timedelta(hours=3)
            while int( current_time.timestamp() ) < int(end_date.timestamp()):
                time.sleep(1)
                current_time = datetime.now(timezone) + timedelta(hours=3)
            print("time of update: {}".format(current_time))


        data_update = terminal.copy_rates_range(symbol=symbol, timeframe=time_frame,
            date_from=update_start_date, date_to=end_date)

        data_update = pd.DataFrame(data_update)
        if len(data_update) == 0:
            return True
        data_update['time'] = pd.to_datetime(data_update['time'], unit='s')
        columns = list(data_update.columns)
        columns[0] = "datetime"
        data_update.columns = columns

        existed_data.datetime = pd.to_datetime( existed_data.datetime, format = "%Y-%m-%d %H:%M:%S" )
        existed_data = existed_data[ existed_data.datetime < data_update.datetime[0] ]

        existed_data = existed_data.append( data_update )
        existed_data.to_csv(self.raw_data_dir + symbol + "_" + time_frame + ".csv", sep="\t", index=False)
        return True