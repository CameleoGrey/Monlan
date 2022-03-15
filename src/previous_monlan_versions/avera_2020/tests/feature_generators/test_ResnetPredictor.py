"""
multi scaler with backtest training and use at test env
"""
from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
from avera.envs.RealCompositeEnv import RealCompositeEnv
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from avera.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from avera.feature_generators.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.terminal.MT5Terminal import MT5Terminal
from avera.mods.HeikenAshiMod import HeikenAshiMod
from avera.mods.EnergyMod import EnergyMod
from avera.mods.VSASpread import VSASpread
import matplotlib.pyplot as plt
from datetime import datetime
from avera.feature_generators.ResnetPredictor import ResnetPredictor


startTime = datetime.now()
print("start time: {}".format(startTime))

#symbol = "EURUSD_i"
symbolList = ["EURUSD_i"]
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "vsa_spread", "tick_volume",
              "hkopen", "hkclose", "enopen", "enclose", "enlow", "enhigh"]

#terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

modDfList = []
for symbol in symbolList:
    #dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2015-01-01 00:00:00")
    df = dataManager.getData(symbol, timeframe)
    df = df.tail(10000)

    ########
    historyMod = VSASpread()
    modDf = historyMod.modHistory(df)
    historyMod = HeikenAshiMod()
    modDf = historyMod.modHistory(modDf)
    historyMod = EnergyMod()
    modDf = historyMod.modHistory(modDf, featList=["open", "close", "low", "high"])
    ########

    modDfList.append(modDf.copy())

resnetPred = ResnetPredictor()
trainX, testX, trainY, testY = resnetPred.makeTrainTestDataSets(dfList=modDfList, nDiffs=1, featureList=hkFeatList)

inputShape = trainX[0].shape
outputShape = trainY[0].shape
resnetPred.build_model(inputShape, outputShape,lr=0.0002)
resnetPred.fit(trainX, trainY, batch_size=10, epochs=20, verbose=1)

resnetPred.save("../../models/", "resnetPredictor")
resnetPred = resnetPred.load("../../models/", "resnet_generator_predictor")

from sklearn.preprocessing import MinMaxScaler, robust_scale, StandardScaler
import numpy as np
#trainY = np.reshape(trainY, (-1, 1))

x = [x for x in range(trainX.shape[0])]
feat = resnetPred.predict(trainX)
#feat = StandardScaler().fit_transform(feat)
feat = feat[:,0]
plt.plot(x, feat, c="g")
#trainY = robust_scale(trainY)
feat = trainY[:,0]
plt.plot(x, feat, c="b")
plt.show()

x = [x for x in range(testX.shape[0])]
feat = resnetPred.predict(testX)
#feat = StandardScaler().fit_transform(feat)
feat = feat[:,0]
plt.plot(x, feat, c="g")
#testY = robust_scale(testY)
feat = testY[:,0]
plt.plot(x, feat, c="b")
plt.show()



priceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList, nDiffs=1, nPoints = 32, flatStack = False, fitOnStep = False)
priceDiffGenerator.setFitMode(True)
priceDiffGenerator = priceDiffGenerator.globalFit(modDf)
priceDiffGenerator.saveGenerator("./MSDiffGen.pkl")

################################
# train agent
################################
openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")

trainDf = modDf.tail(6040).head(5040)
trainEnv = CompositeEnv(trainDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001, renderFlag=True)
#backTestDf = modDf.tail(7200).head(1200)
backTestDf = modDf.tail(6040).tail(1040)
backTestEnv = CompositeEnv(backTestDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001, renderFlag=True)
# get size of state and action from environment
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=2500, batch_size=100, train_start=200, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=0.9999, learning_rate=0.0002)
buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                          epsilon_decay=1.0, learning_rate=0.0002)
sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=5000, batch_size=200, train_start=300, epsilon_min=0.05, epsilon=1,
                           epsilon_decay=1.0, learning_rate=0.0002)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
#agent  = agent.load_agent("./", "best_composite")
#agent  = agent.load_agent("./", "checkpoint_composite")
lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=15, nWarmUp=0,
                    uniformEps=False, synEps=True, plotScores=True, saveBest=True, saveFreq=1)

endTime = datetime.now()
print("Training finished. Total time: {}".format(endTime - startTime))


###############################
# use agent
###############################

#testDf = modDf.tail(7200).head(1200)
testDf = modDf.tail(1000)

openerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
buyerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")
sellerPriceDiffGenerator = MultiScalerDiffGenerator(featureList=hkFeatList).loadGenerator("./MSDiffGen.pkl")

#openerPriceDiffGenerator.setFitMode(False)
#buyerPriceDiffGenerator.setFitMode(False)
#sellerPriceDiffGenerator.setFitMode(False)

testEnv = CompositeEnv(testDf, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001, renderFlag=True)

# get size of state and action from environment
openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent  = agent.load_agent("./", "best_composite", dropSupportModel=True)
#agent  = agent.load_agent("./", "checkpoint_composite")
print("start using agent")

dealsStatistics = agent.use_agent(testEnv)
sumRew = 0
cumulativeReward = []
for i in range(len(dealsStatistics)):
    sumRew += dealsStatistics[i]
    cumulativeReward.append(sumRew)
plt.plot( [x for x in range(len(cumulativeReward))], cumulativeReward )
plt.show()