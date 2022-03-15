from avera.agents.DQNAgent import DQNAgent
from avera.agents.CompositeAgent import CompositeAgent
from avera.envs.CompositeEnv import CompositeEnv
from avera.envs.RealCompositeEnv import RealCompositeEnv
#from avera.feature_generators.FeatureScaler import FeatureScaler
from avera.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from avera.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from avera.feature_generators.W2VDiffGenerator import W2VDiffGenerator
from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from avera.terminal.MT5Terminal import MT5Terminal
import matplotlib.pyplot as plt
import joblib

def getTrainTestSets(df, nExperiments, mainPackSize=2044, trainSize=0.5):
    trainTestSets = []
    dataStep = df.shape[0] // nExperiments
    for i in range(nExperiments):
        dataPart = df[i*dataStep : (i+1)*dataStep]
        dataPart = dataPart.tail(mainPackSize)
        trainSet = dataPart[ : int(trainSize * dataPart.shape[0]) ]
        testSet = dataPart[ int(trainSize * dataPart.shape[0]) : ]
        trainTestSets.append( [trainSet, testSet] )
    return trainTestSets

def createGenerators(df, priceFeatList, volumeFeatList):
    priceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList, nDiffs=1, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=3, iter=100, min_count=0, sample=0.0, sg=0)

    volumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList, nDiffs=1, nPoints = 21, flatStack = False, fitOnStep = False,
                 nIntervals = 10000, w2vSize=32, window=3, iter=100, min_count=0, sample=0.0, sg=0)


    priceDiffGenerator.setFitMode(True)
    priceDiffGenerator = priceDiffGenerator.globalFit(df)
    priceDiffGenerator.saveGenerator("./w2vPriceDiffGen.pkl")

    volumeDiffGenerator.setFitMode(True)
    volumeDiffGenerator = volumeDiffGenerator.globalFit(df)
    volumeDiffGenerator.saveGenerator("./w2vVolumeDiffGen.pkl")
    pass

def trainAgent(trainDf, startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001 ):
    openerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
    buyerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
    sellerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")

    openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
    buyerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
    sellerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")


    openerCompositeGenerator = W2VCompositeGenerator( [openerPriceDiffGenerator,openerVolumeDiffGenerator], flatStack=False)
    buyerCompositeGenerator = W2VCompositeGenerator( [buyerPriceDiffGenerator,buyerVolumeDiffGenerator], flatStack=False)
    sellerCompositeGenerator = W2VCompositeGenerator( [sellerPriceDiffGenerator,sellerVolumeDiffGenerator], flatStack=False)

    trainEnv = CompositeEnv(trainDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=startDeposit, lotSize=lotSize, lotCoef=lotCoef, spread = spread, spreadCoef = spreadCoef, renderFlag=True)
    openerAgent = DQNAgent(trainEnv.observation_space["opener"], trainEnv.action_space["opener"].n,
                        memorySize=500, batch_size=100, train_start=200, epsilon_min=0.1, epsilon=1, epsilon_decay=0.9994)
    buyerAgent = DQNAgent(trainEnv.observation_space["buyer"], trainEnv.action_space["buyer"].n,
                        memorySize=1000, batch_size=200, train_start=300, epsilon_min=0.1, epsilon=1, epsilon_decay=0.9999)
    sellerAgent = DQNAgent(trainEnv.observation_space["seller"], trainEnv.action_space["seller"].n,
                        memorySize=1000, batch_size=200, train_start=300, epsilon_min=0.1, epsilon=1, epsilon_decay=0.9999)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    agent.fit_agent(env=trainEnv, nEpisodes=15, nWarmUp = 0, uniformEps = False, plotScores=True, saveBest=True, saveFreq=2)

def collectStatistics(testDf, startDeposit=300, lotSize=0.1, lotCoef=100000, spread=18, spreadCoef=0.00001 ):
    openerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
    buyerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")
    sellerPriceDiffGenerator = W2VDiffGenerator(featureList=priceFeatList).loadGenerator("./w2vPriceDiffGen.pkl")

    openerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
    buyerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")
    sellerVolumeDiffGenerator = W2VDiffGenerator(featureList=volumeFeatList).loadGenerator("./w2vVolumeDiffGen.pkl")

    openerPriceDiffGenerator.setFitMode(False)
    buyerPriceDiffGenerator.setFitMode(False)
    sellerPriceDiffGenerator.setFitMode(False)

    openerVolumeDiffGenerator.setFitMode(False)
    buyerVolumeDiffGenerator.setFitMode(False)
    sellerVolumeDiffGenerator.setFitMode(False)

    openerCompositeGenerator = W2VCompositeGenerator( [openerPriceDiffGenerator,openerVolumeDiffGenerator], flatStack=False)
    buyerCompositeGenerator = W2VCompositeGenerator( [buyerPriceDiffGenerator,buyerVolumeDiffGenerator], flatStack=False)
    sellerCompositeGenerator = W2VCompositeGenerator( [sellerPriceDiffGenerator,sellerVolumeDiffGenerator], flatStack=False)

    testEnv = CompositeEnv(testDf, openerCompositeGenerator, buyerCompositeGenerator, sellerCompositeGenerator,
                        startDeposit=startDeposit, lotSize=lotSize, lotCoef=lotCoef, spread = spread, spreadCoef = spreadCoef, renderFlag=True)
    openerAgent = DQNAgent(testEnv.observation_space["opener"], testEnv.action_space["opener"].n)
    buyerAgent = DQNAgent(testEnv.observation_space["buyer"], testEnv.action_space["buyer"].n)
    sellerAgent = DQNAgent(testEnv.observation_space["seller"], testEnv.action_space["seller"].n)
    agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent)
    agent  = agent.load_agent("./", "best_composite")
    print("start using agent")
    dealsStatistics = agent.use_agent(testEnv)
    return dealsStatistics
#########################################################################################################
timeframe = "M15"
symbolList = ["GBPCHF_i", "NZDJPY_i", "AUDCAD_i", "EURUSD_i"]
priceFeatList = ["open", "close", "low", "high"]
volumeFeatList = ["tick_volume"]
spreadDict = { "GBPCHF_i": 83, "NZDJPY_i": 37, "AUDCAD_i": 83, "EURUSD_i": 18 }
spreadCoefDict = { "GBPCHF_i": 0.00001, "NZDJPY_i": 0.001, "AUDCAD_i": 0.00001, "EURUSD_i": 0.00001 }
terminal = MT5Terminal()
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")
###########################################################################################################

for symbol in symbolList:
    dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2016-01-01 00:00:00")
    df = SymbolDataManager("../../data/raw/").getData(symbol, timeframe)
    createGenerators(df, priceFeatList, volumeFeatList)

    trainTestSets = getTrainTestSets(df, nExperiments=10, mainPackSize=2044, trainSize=0.5)
    trainDealsStatistics = []
    testDealsStatistics = []
    nStep = 0
    for trainX, testX in trainTestSets:
        nStep += 1
        print("Experiment №{}".format(nStep))
        trainAgent(trainX.copy(), spread=spreadDict[symbol], spreadCoef=spreadCoefDict[symbol])
        trainDealsStatistics.append( collectStatistics(trainX, spread=spreadDict[symbol], spreadCoef=spreadCoefDict[symbol]) )
        testDealsStatistics.append( collectStatistics(testX, spread=spreadDict[symbol], spreadCoef=spreadCoefDict[symbol]) )
        with open("./" + "trainDealsStatistics_{}_{}.pkl".format(symbol, timeframe), mode="wb") as dealsFile:
            joblib.dump(trainDealsStatistics, dealsFile)
        with open("./" + "testDealsStatistics_{}_{}.pkl".format(symbol, timeframe), mode="wb") as dealsFile:
            joblib.dump(testDealsStatistics, dealsFile)
print("statistics collected")

"""trainDealsStatistics = None
testDealsStatistics = None
with open("./" + "trainDealsStatistics_{}_{}.pkl".format(symbol, timeframe), mode="rb") as dealsFile:
    trainDealsStatistics = joblib.load(dealsFile)
with open("./" + "dealsStatistics_{}_{}.pkl".format(symbol, timeframe), mode="rb") as dealsFile:
    testDealsStatistics = joblib.load(dealsFile)

for j in range(len(trainDealsStatistics)):
    sumRew = 0
    cumulativeReward = []
    for i in range(len(trainDealsStatistics[j])):
        sumRew += trainDealsStatistics[j][i]
        cumulativeReward.append(sumRew)
    plt.plot( [x for x in range(len(cumulativeReward))], cumulativeReward )
plt.show()

for j in range(len(testDealsStatistics)):
    sumRew = 0
    cumulativeReward = []
    for i in range(len(testDealsStatistics[j])):
        sumRew += testDealsStatistics[j][i]
        cumulativeReward.append(sumRew)
    plt.plot( [x for x in range(len(cumulativeReward))], cumulativeReward )
plt.show()"""
