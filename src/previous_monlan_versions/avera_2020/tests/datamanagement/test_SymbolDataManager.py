
from avera.datamanagement.SymbolDataManager import SymbolDataManager

symbolDM = SymbolDataManager()

readedData = symbolDM.getData(symbol="TEST", timeFrame="H1")
print(readedData.tail(2))
readedData = symbolDM.getData(symbol="TEST", timeFrame="H1", normalizeNames=True)
print(readedData.tail(2))
readedData = symbolDM.getData(symbol="TEST", timeFrame="H1", normalizeNames=True, normalizeDateTime=True)
print(readedData.tail(2))