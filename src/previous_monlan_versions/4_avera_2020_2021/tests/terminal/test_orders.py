
from avera.terminal.MT5Terminal import MT5Terminal
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from datetime import datetime, timedelta
from dateutil import parser
import time

terminal = MT5Terminal()
symbolUpdater = SymbolDataUpdater()
symbolManager = SymbolDataManager()

symbol = "EURUSD_i"
timeframe = "M5"

symbolUpdater.partialUpdate(terminal, symbol, timeframe)
rawData = symbolManager.getData(symbol, timeframe)
lastRow = rawData.tail(1)
lastDateTime = parser.parse(lastRow["datetime"].values[0])
while (True):
    symbolUpdater.partialUpdate(terminal, symbol, timeframe)
    rawData = symbolManager.getData(symbol, timeframe)
    lastRow = rawData.tail(1)
    freshDateTime = parser.parse(lastRow["datetime"].values[0])
    if freshDateTime != lastDateTime:
        lastDateTime = freshDateTime
        break
    else:
        time.sleep(10)
# place order (buy/sell by market)
orderType = "sell"
lotSize = 0.01
order = terminal.placeOrder( orderType, symbol, lotSize )
#update data each 10 sec, if last datetime is different then stop waiting
time.sleep(1)
#close order
terminal.closeOrder( order )