
from monlan.agents.DQNAgent import DQNAgent
from monlan.agents.CompositeAgent import CompositeAgent
from monlan.envs.CompositeEnv import CompositeEnv
from monlan.envs.RealCompositeEnv import RealCompositeEnv
#from monlan.feature_generators.FeatureScaler import FeatureScaler
from monlan.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from monlan.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from monlan.feature_generators.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from monlan.datamanagement.SymbolDataManager import SymbolDataManager
from monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from monlan.terminal.MT5Terminal import MT5Terminal
from monlan.mods.HeikenAshiMod import HeikenAshiMod
from monlan.agents.AlwaysHoldAgent import AlwaysHoldAgent
from monlan.mods.EnergyMod import EnergyMod
from monlan.mods.VSASpread import VSASpread
import matplotlib.pyplot as plt
from datetime import datetime
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "spread", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
#dataUpdater = SymbolDataUpdater("../data/raw/")
dataManager = SymbolDataManager("../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
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

priceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList, nDiffs=1, nPoints = 256, flatStack = False, fitOnStep = False)
priceDiffGenerator.setFitMode(True)
priceDiffGenerator = priceDiffGenerator.globalFit(modDf)
priceDiffGenerator.saveGenerator("../models/MSDiffGen.pkl")

###############################
# use agent
###############################
#from monlan.utils.keras_utils import reset_keras
#reset_keras()

#testDf = modDf.head(400000)
#testDf = modDf.tail(12000).head(2000)
#testDf = modDf.tail(10000).head(8000)
#testDf = modDf.tail(int(0.05 * len(modDf)))
testDf = modDf.tail(200000).head(100000).tail(20000)
#testDf = modDf.tail(110000).head(10000) #back

openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("../models/MSDiffGen.pkl")
buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("../models/MSDiffGen.pkl")
sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("../models/MSDiffGen.pkl")

#openerPriceDiffGenerator.setFitMode(False)
#buyerPriceDiffGenerator.setFitMode(False)
#sellerPriceDiffGenerator.setFitMode(False)

testEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                       startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001,
                       stopType="adaptive", takeType="adaptive", stopPos=1, takePos=1, maxLoss=20000, maxTake=20000,
                       stoplossPuncts=2000, takeprofitPuncts=2000, riskPoints=110, riskLevels=5, parallelOpener=False,
                       renderDir="../models/", renderName="test_plot")

# get size of state and action from environment
openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n, fillMemoryByPretrainedModel=True)
buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n, fillMemoryByPretrainedModel=True)
sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n, fillMemoryByPretrainedModel=True)
#buyerAgent = AlwaysHoldAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n, agentName="buyer")
#sellerAgent = AlwaysHoldAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n, agentName="seller")
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
#agent = agent.load_agent("../models/", "best_composite", dropSupportModel=True)
#agent = agent.loadPretrainedAgents(dir="../models/", baseName="super_resnet_34_{}_{}".format(symbol, timeframe))
agent = agent.loadPretrainedAgents(dir="../models/", baseName="qrl_resnet_34")
#agent  = agent.load_agent("../models/", "checkpoint_composite")
print("start using agent")

dealsStatistics = agent.use_agent(testEnv)

###########################
import numpy as np
dealAvg = np.sum(dealsStatistics) / len(dealsStatistics)
dealStd = np.std(dealsStatistics)
print("Avg deal profit: {}".format(dealAvg))
print("Deal's std: {}".format(dealStd))
###########################

sumRew = 0
cumulativeReward = []
for i in range(len(dealsStatistics)):
    sumRew += dealsStatistics[i]
    cumulativeReward.append(sumRew)
plt.plot( [x for x in range(len(cumulativeReward))], cumulativeReward )
plt.show()