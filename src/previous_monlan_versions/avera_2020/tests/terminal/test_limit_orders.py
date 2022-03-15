
from avera.terminal.MT5Terminal import MT5Terminal
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from datetime import datetime, timedelta
from dateutil import parser
import time

terminal = MT5Terminal()
symbolUpdater = SymbolDataUpdater()
symbolManager = SymbolDataManager()

symbol = "GBPUSD_i"
timeframe = "M5"

# place order (buy/sell by market)
orderType = "sell"
lotSize = 0.01
order = terminal.placeOrder( orderType, symbol, lotSize, stoplossPuncts=50, takeprofitPuncts=50 )
#update data each 10 sec, if last datetime is different then stop waiting
time.sleep(600)
trigger = terminal.checkLimitTrigger(symbol)
#close order
terminal.closeOrder( order )