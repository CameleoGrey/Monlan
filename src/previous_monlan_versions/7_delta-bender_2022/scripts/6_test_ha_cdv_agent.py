
from legacy.classes.monlan.agents.CompositeAgent import CompositeAgent
from legacy.classes.monlan.envs.CompositeEnv import CompositeEnv
from legacy.classes.monlan.datamanagement.SymbolDataManager import SymbolDataManager

from classes.delta_bender.FeatGen_Ident import FeatGen_Ident
from classes.delta_bender.ResnetBuilder import ResnetBuilder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from classes.delta_bender.agents.DBAgentOpener_HACDV import DBAgentOpener_HACDV
from classes.delta_bender.agents.DBAgentBuyer_HACDV import DBAgentBuyer_HACDV
from classes.delta_bender.agents.DBAgentSeller_HACDV import DBAgentSeller_HACDV

import matplotlib.pyplot as plt
from datetime import datetime
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
#dataUpdater = SymbolDataUpdater("../data/raw/")
dataManager = SymbolDataManager("../data/raw/")
#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = dataManager.getData(symbol, timeframe)
#df = df.tail(int(0.95*(len(df))))

#open = df["open"].values
#high = df["high"].values
#low = df["low"].values
#close = df["close"].values

df = df.tail(3000)

target_cols = ["open", "high", "low", "close", "cdv"]
openerPriceDiffGenerator = FeatGen_Ident(target_cols, nPoints=200)
buyerPriceDiffGenerator = FeatGen_Ident(target_cols, nPoints=200)
sellerPriceDiffGenerator = FeatGen_Ident(target_cols, nPoints=200)

testEnv = CompositeEnv(df, openerPriceDiffGenerator, buyerPriceDiffGenerator, sellerPriceDiffGenerator,
                       startDeposit=300, lotSize=0.1, lotCoef=100000, spread=23, spreadCoef=0.00001,
                       stopType="std", takeType="std", stopPos=2, takePos=2, maxLoss=2000, maxTake=80,
                       stoplossPuncts=2000, takeprofitPuncts=200, riskPoints=64, riskLevels=4, parallelOpener=False,
                       renderDir="../models/", renderName="test_plot")


model = ResnetBuilder().build_resnet_34( (1, 8, 64), num_outputs=2, outputActivation="softmax" )
model.compile( optimizer=Adam(learning_rate=0.0001), loss=CategoricalCrossentropy() )
model.load_weights( "../models/resnet_ha_cake_softmax" + ".h5" )

openerAgent = DBAgentOpener_HACDV(agentName="opener", model=model)
buyerAgent = DBAgentBuyer_HACDV(agentName="buyer", model=model)
sellerAgent = DBAgentSeller_HACDV(agentName="seller", model=model)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
print("start using agent")


print(df.shape)
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