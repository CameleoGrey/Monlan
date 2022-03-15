
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.terminal.MT5Terminal import MT5Terminal

symUpdater = SymbolDataUpdater()
terminal = MT5Terminal()
#symUpdater.fullUpdate( terminal=terminal, symbol="EURUSD_i", timeFrame="H1", startDate="2020-05-01 00:00:00")
symUpdater.partialUpdate(terminal=terminal, symbol="EURUSD_i", timeFrame="H1")
