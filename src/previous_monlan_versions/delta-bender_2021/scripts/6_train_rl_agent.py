
from legacy.classes.monlan.agents.DQNAgent import DQNAgent
from legacy.classes.monlan.agents.CompositeAgent import CompositeAgent
from legacy.classes.monlan.envs.CompositeEnv import CompositeEnv
from legacy.classes.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from classes.delta_bender.FeatGenForRL import FeatGenForRL
import matplotlib.pyplot as plt
from datetime import datetime
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M15"
target_columns = [ "datetime", "open", "close", "low", "high", "spread", "tick_volume"]


dataManager = SymbolDataManager("../data/raw/")

df = dataManager.getData(symbol, timeframe)

feat_gen = FeatGenForRL(featureList=target_columns, nPoints = 128, window_size = 64,
                                                    ema_period_volume = 14, ema_period_price = 3)

################################
# train agent
################################

trainDf = df.tail(72000).head(60000)
trainEnv = CompositeEnv(trainDf, feat_gen, feat_gen, feat_gen,
                        startDeposit=300, lotSize=0.1, lotCoef=100000, spread=23, spreadCoef=0.00001,
                        stopType="const", takeType="const", stopPos=3, takePos=3, maxLoss=2000, maxTake=2000,
                        stoplossPuncts=20000, takeprofitPuncts=20000, riskPoints=64, riskLevels=5, parallelOpener=False,
                        renderFlag=True, renderDir="../models/", renderName="train_plot")
backTestDf = df.tail(84000).head(12000)
#backTestDf = df.head(50000).tail(3192).head(1192)
backTestEnv = CompositeEnv(backTestDf, feat_gen, feat_gen, feat_gen,
                           startDeposit=300, lotSize=0.1, lotCoef=100000, spread=23, spreadCoef=0.00001,
                           stopType="adaptive", takeType="adaptive", stopPos=3, takePos=3, maxLoss=2000, maxTake=2000,
                           stoplossPuncts=20000, takeprofitPuncts=20000, riskPoints=64, riskLevels=5, parallelOpener=False,
                           renderDir="../models/", renderName="back_plot")
# get size of state and action from environment
openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                           memorySize=32000, batch_size=16, train_start=31000, epsilon_min=0.2, epsilon=1, discount_factor=0.9995,
                           epsilon_decay=0.9999, learning_rate=0.0001)
buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                          memorySize=32000, batch_size=16, train_start=16000, epsilon_min=0.2, epsilon=1, discount_factor=0.9995,
                          epsilon_decay=0.9999, learning_rate=0.0001)
sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                           memorySize=32000, batch_size=16, train_start=16000, epsilon_min=0.2, epsilon=1, discount_factor=0.9995,
                           epsilon_decay=0.9999, learning_rate=0.0001)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)

###################################
lastSaveEp = agent.fit_agent(env=trainEnv, backTestEnv=backTestEnv, nEpisodes=200, nWarmUp=0,
                             uniformEps=False, synEps=False, plotScores=False,
                             saveBest=True, saveFreq=1, saveDir="../models/", saveName="best_composite")

endTime = datetime.now()
print("Training finished. Total time: {}".format(endTime - startTime))

###############################
# use agent
###############################

#testDf = df.tail(12000).head(2000)
#testDf = df.tail(10000).head(8000)
testDf = df.tail(72000).tail(12000)
#testDf = df.tail(110000).head(10000) #back


testEnv = CompositeEnv(testDf, feat_gen, feat_gen, feat_gen,
                       startDeposit=300, lotSize=0.1, lotCoef=100000, spread=23, spreadCoef=0.00001,
                       stopType="adaptive", takeType="adaptive", stopPos=2, takePos=2, maxLoss=20000, maxTake=20000,
                       stoplossPuncts=20000, takeprofitPuncts=20000, riskPoints=64, riskLevels=5, parallelOpener=False,
                       renderDir="../models/", renderName="test_plot")

# get size of state and action from environment
openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
agent  = agent.load_agent("../models/", "best_composite", dropSupportModel=True)
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