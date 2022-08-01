
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
from monlan.utils.save_load import *
import gc


startTime = datetime.now()
print("start time: {}".format(startTime))

symbols = ["EURUSD_i"]
#symbols = ["AUDUSD_i", "EURAUD_i", "USDCAD_i", "EURCAD_i", "GBPUSD_i",]
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "spread", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
#dataUpdater = SymbolDataUpdater("../data/raw/")
dataManager = SymbolDataManager("../data/raw/")

#for symbol in symbols:
#    print("Update data for {}".format(symbol))
#    dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2010-01-01 00:00:00")

for symbol in symbols:
    df = dataManager.getData(symbol, timeframe)
    df = df.tail(400000)

    ########
    historyMod = VSASpread()
    modDf = historyMod.modHistory(df)
    historyMod = HeikenAshiMod()
    modDf = historyMod.modHistory(modDf)
    historyMod = EnergyMod()
    modDf = historyMod.modHistory(modDf, featList=["open", "close", "low", "high"])
    ########
    del df
    gc.collect()

    priceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList, nDiffs=1, nPoints = 256, flatStack = False, fitOnStep = False)
    priceDiffGenerator.setFitMode(True)
    priceDiffGenerator = priceDiffGenerator.globalFit(modDf)
    priceDiffGenerator.saveGenerator("../models/opener_MSDiffGen.pkl")

    sampleGen = SamplesGenerator(priceDiffGenerator, lotSize=0.1, lotCoef=100000, spreadCoef = 0.00001)

    #x_train, y_train = sampleGen.generateOpenerSamples(modDf, windowSize = 10)
    #save("../data/train_samples/opener_x_{}_{}.pkl".format(symbol, timeframe), x_train)
    #save("../data/train_samples/opener_y_{}_{}.pkl".format(symbol, timeframe), y_train)
    #del x_train, y_train
    #gc.collect()

    x_train, y_train = sampleGen.generateBuyerSamples(modDf, windowSize = 10)
    save("../data/train_samples/buyer_x_{}_{}.pkl".format(symbol, timeframe), x_train)
    save("../data/train_samples/buyer_y_{}_{}.pkl".format(symbol, timeframe), y_train)
    del x_train, y_train
    gc.collect()

    x_train, y_train = sampleGen.generateSellerSamples(modDf, windowSize = 10)
    save("../data/train_samples/seller_x_{}_{}.pkl".format(symbol, timeframe), x_train)
    save("../data/train_samples/seller_y_{}_{}.pkl".format(symbol, timeframe), y_train)
    del x_train, y_train
    gc.collect()

    del modDf
    gc.collect()

    print("done")
