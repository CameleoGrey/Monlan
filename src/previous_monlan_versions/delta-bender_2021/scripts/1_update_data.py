from classes.delta_bender.MT5Terminal import MT5Terminal
from classes.delta_bender.SymbolDataUpdater import SymbolDataUpdater
from classes.delta_bender.SymbolDataManager import SymbolDataManager

symbol = "EURUSD_i"
timeframe = "M30"

terminal = MT5Terminal(login=123456, server="broker-server", password="password")
dataUpdater = SymbolDataUpdater("../data/raw/")
dataManager = SymbolDataManager("../data/raw/")

dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
#dataUpdater.partialUpdate(terminal, symbol, timeframe)
df = dataManager.getData(symbol, timeframe)
print(df)
print("done")