"""
ResNet + MultiScaler + extended rows
"""
from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
from avera.feature_generators.CompositeGenerator import CompositeGenerator
from avera.feature_generators.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.terminal.MT5Terminal import MT5Terminal
from avera.mods.HeikenAshiMod import HeikenAshiMod
from avera.mods.EnergyMod import EnergyMod
from avera.mods.VSASpread import VSASpread
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.linear_model import HuberRegressor

startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "H1"
hkFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]

dataManager = SymbolDataManager()
df = SymbolDataManager().getData(symbol, timeframe)
df = df.tail(10000)

########
historyMod = HeikenAshiMod()
modDf = historyMod.modHistory(df)
########

nIntervals = 100
nLevels = 8
featVals = modDf["hkclose"].values
minVal = np.min(featVals)
maxVal = np.max(featVals)
step = (maxVal - minVal) / nIntervals
borders = []
intervalDict = {}
for i in range(nIntervals):
    border = minVal + i * step
    borders.append( border )
    intervalDict[border] = [[], []]
borders = np.asarray( borders )

for i in range(featVals.shape[0]):
    featVal = featVals[i]
    idx = (np.abs(borders - featVal)).argmin()
    intervalDict[borders[idx]][0].append(i)
    intervalDict[borders[idx]][1].append(featVal)

levels = intervalDict.items()
levels = list(sorted(levels, key=lambda l: len(l[1][0]), reverse=True))

levelBuilders = []
for i in range(nLevels):
    level = levels[i]
    x = np.asarray(level[1][0])
    x = np.reshape(x, (-1, 1))
    y = np.asarray(level[1][1])
    y = np.reshape(y, (-1,))
    linReg = HuberRegressor()
    linReg.fit(x, y)
    levelBuilders.append(linReg)

for linReg in levelBuilders:
    x = np.asarray([x for x in range(featVals.shape[0])])
    x = np.reshape(x, (-1, 1))
    levelVals = linReg.predict(x)
    x = np.reshape(x, (-1,))
    plt.plot(x, levelVals)

levelBuilders = []
levels = levels[:int(nIntervals * 0.5)]
for i in range(nLevels):
    level = levels[-i]
    x = np.asarray(level[1][0])
    x = np.reshape(x, (-1, 1))
    y = np.asarray(level[1][1])
    y = np.reshape(y, (-1,))
    linReg = HuberRegressor()
    linReg.fit(x, y)
    levelBuilders.append(linReg)

for linReg in levelBuilders:
    x = np.asarray([x for x in range(featVals.shape[0])])
    x = np.reshape(x, (-1, 1))
    levelVals = linReg.predict(x)
    x = np.reshape(x, (-1,))
    plt.plot(x, levelVals)

plt.plot([x for x in range(len(featVals))], featVals)
plt.show()

