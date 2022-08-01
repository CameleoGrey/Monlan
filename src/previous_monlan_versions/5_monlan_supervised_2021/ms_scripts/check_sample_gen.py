
from monlan.envs.CompositeEnv import CompositeEnv
from monlan.envs.RealCompositeEnv import RealCompositeEnv
from monlan.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from monlan.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from monlan_supervised.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from monlan.datamanagement.SymbolDataManager import SymbolDataManager
from monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from monlan.terminal.MT5Terminal import MT5Terminal
from monlan.mods.HeikenAshiMod import HeikenAshiMod
from monlan.mods.EnergyMod import EnergyMod
from monlan.mods.VSASpread import VSASpread
import matplotlib.pyplot as plt
from datetime import datetime
from monlan_supervised.SamplesGenerator import SamplesGenerator
import numpy as np


startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "tick_volume"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
#dataUpdater = SymbolDataUpdater("../data/raw/")
dataManager = SymbolDataManager("../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = dataManager.getData(symbol, timeframe)
df = df.tail(1043)

########
"""historyMod = VSASpread()
modDf = historyMod.modHistory(df)
historyMod = HeikenAshiMod()
modDf = historyMod.modHistory(modDf)
historyMod = EnergyMod()
modDf = historyMod.modHistory(modDf, featList=["open", "close", "low", "high"])"""
########

priceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList, nDiffs=1, nPoints = 32, flatStack = False, fitOnStep = False)
priceDiffGenerator.setFitMode(True)
priceDiffGenerator = priceDiffGenerator.globalFit(df)
priceDiffGenerator.saveGenerator("../models/MSDiffGen.pkl")

sampleGen = SamplesGenerator(priceDiffGenerator)
x_train, y_train = sampleGen.generateOpenerSamples(df, windowSize = 10)
print(x_train.shape)
print(np.mean(y_train, axis=0))
print("done")
