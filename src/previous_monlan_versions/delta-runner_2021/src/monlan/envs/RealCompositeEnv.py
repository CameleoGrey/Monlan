import matplotlib.pyplot as plt
from copy import deepcopy
from dateutil import parser
from src.monlan.agents.RiskManager import RiskManager
import time

class RealCompositeEnv():
    class OpenerActionSpace():
        def __init__(self):
            self.actionsDict = { 0: "buy", 1: "hold", 2: "sell" }
            self.n = 3
            pass
    class BuyerActionSpace():
        def __init__(self):
            self.actionsDict = { 0: "hold", 1: "buy" }
            self.n = 2
            pass
    class SellerActionSpace():
        def __init__(self):
            self.actionsDict = { 0: "hold", 1: "sell" }
            self.n = 2
            pass

    def __init__(self, symbol, timeframe, terminal, dataUpdater, dataManager,
                 openerFeatureGen, buyerFeatureGen, sellerFeatureGen,
                 startDeposit=300, lotSize=0.1, lotCoef=100000, stopType="adaptive", takeType="adaptive",
                 stoplossPuncts=100, takeprofitPuncts=500, stopPos=2, takePos=1, maxLoss=40000, maxTake=40000,
                 riskPoints=110, riskLevels=5, renderFlag=False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.terminal = terminal
        self.stoplossPuncts = stoplossPuncts
        self.takeprofitPuncts = takeprofitPuncts
        self.dataUpdater = dataUpdater
        self.dataManager = dataManager
        self.historyPrice = None
        self.done = False
        self.startDeposit = startDeposit
        self.deposit = self.startDeposit
        self.lotSize = lotSize
        self.lotCoef = lotCoef
        self.renderFlag = renderFlag
        self.stopType = stopType
        self.takeType = takeType
        self.stoplossPuncts = stoplossPuncts
        self.takeprofitPuncts = takeprofitPuncts
        self.stopPos = stopPos
        self.takePos = takePos
        self.maxLoss = maxLoss
        self.maxTake = maxTake
        self.riskManager = RiskManager(nPoints=riskPoints, nLevels=riskLevels,
                                       stopPos=stopPos, takePos=takePos,
                                       spreadCoef=1 / self.lotCoef)

        self.openData = []
        self.closeData = []
        self.xData = []
        self.iStep = 0
        self.order = None

        self.actorNames = ["opener", "buyer", "seller"]
        self.featureGenerators = {"opener": openerFeatureGen,
                                  "buyer": buyerFeatureGen,
                                  "seller": sellerFeatureGen}
        self.action_space = {"opener": self.OpenerActionSpace(),
                             "buyer": self.BuyerActionSpace(),
                             "seller": self.SellerActionSpace()}
        self.observation_space = {"opener": openerFeatureGen.featureShape,
                                  "buyer": buyerFeatureGen.featureShape,
                                  "seller": sellerFeatureGen.featureShape}
        self.openPoint = None
        self.mode = None
        self.sumLoss = 0

        pass

    def reset(self):
        self.setMode("opener")
        self.openPoint = None
        self.sumLoss = 0
        self.done = False
        self.deposit = self.startDeposit
        self.openData = []
        self.closeData = []
        self.xData = []
        self.iStep = 0
        self.order = None
        if self.renderFlag == True:
            plt.close()

        self.updateHistoryPrice()
        #start from next bar
        #self.getNextBar(checkTime=30)
        lastDate = self.extractLastDate(self.historyPrice)

        obs = self.featureGenerators[self.mode].getFeatByDatetime(lastDate, self.historyPrice)

        return obs

    def step(self, action):
        info = {}
        reward = None
        action = self.action_space[self.mode].actionsDict[action]
        self.iStep += 1

        if self.mode == "opener":
            nextHistoryRow = next( self.historyPrice.tail(1).iterrows() )
            nextDate = nextHistoryRow[0]
            if action == "buy":
                if self.stopType == "adaptive" or self.takeType == "adaptive":
                    limitsDict = self.riskManager.getAdaptiveLimits(nextHistoryRow[0], self.historyPrice, nextHistoryRow[1]["open"])
                    self.stoplossPuncts = limitsDict["stop"]
                    self.takeprofitPuncts = limitsDict["take"]
                    if self.stoplossPuncts > self.maxLoss:
                        self.stoplossPuncts = self.maxLoss
                    if self.takeprofitPuncts > self.maxTake:
                        self.takeprofitPuncts = self.maxTake
                self.order = self.terminal.placeOrder(action, self.symbol, self.lotSize, self.stoplossPuncts, self.takeprofitPuncts)
                #nextHistoryRow = self.getNextBar(checkTime=30)
                #nextDate = nextHistoryRow[0]
                self.mode = "buyer"
                self.openPoint = deepcopy(nextHistoryRow)
                self.sumLoss = 0
                reward = 0
            elif action == "sell":
                if self.stopType == "adaptive" or self.takeType == "adaptive":
                    limitsDict = self.riskManager.getAdaptiveLimits(nextHistoryRow[0], self.historyPrice, nextHistoryRow[1]["open"])
                    self.stoplossPuncts = limitsDict["stop"]
                    self.takeprofitPuncts = limitsDict["take"]
                    if self.stoplossPuncts > self.maxLoss:
                        self.stoplossPuncts = self.maxLoss
                    if self.takeprofitPuncts > self.maxTake:
                        self.takeprofitPuncts = self.maxTake
                self.order = self.terminal.placeOrder(action, self.symbol, self.lotSize, self.stoplossPuncts, self.takeprofitPuncts)
                #nextHistoryRow = self.getNextBar(checkTime=30)
                #nextDate = nextHistoryRow[0]
                self.mode = "seller"
                self.openPoint = deepcopy(nextHistoryRow)
                self.sumLoss = 0
                reward = 0
            elif action == "hold":
                self.sumLoss += 0
                reward = self.sumLoss
                nextHistoryRow = self.getNextBar(checkTime=10)
                nextDate = nextHistoryRow[0]

            obs = self.featureGenerators[self.mode].getFeatByDatetime(nextDate, self.historyPrice)
            if self.renderFlag == True:
                self.render(self.iStep, nextHistoryRow[1]["open"], nextHistoryRow[1]["close"], action)
            return obs, reward, self.done, info

        nextHistoryRow = self.getNextBar(checkTime=10)
        nextDate = nextHistoryRow[0]
        if self.mode in ["buyer", "seller"]:
            if action == "hold":
                obs = self.featureGenerators[self.mode].getFeatByDatetime(nextDate, self.historyPrice)
                if self.renderFlag == True:
                    self.render(self.iStep, nextHistoryRow[1]["open"], nextHistoryRow[1]["close"], action)
                rewardDict = {}
                if self.mode == "buyer":
                        rewardDict[0] = 0
                        rewardDict[1] = 0
                elif self.mode == "seller":
                        rewardDict[0] = 0
                        rewardDict[1] = 0
                info["limitOrder"] = False
                limitTrigger = self.terminal.checkLimitTrigger(self.symbol)
                if limitTrigger:
                    print("Stop/take order triggered.")
                    self.mode = "opener"
                    nextOpenerObs = self.featureGenerators[self.mode].getFeatByDatetime(nextDate, self.historyPrice)
                    info["nextOpenerState"] = nextOpenerObs
                    info["limitOrder"] = True

                return obs, rewardDict, self.done, info

            saveMode = self.mode
            nextCloserObs = self.featureGenerators[self.mode].getFeatByDatetime(nextDate, self.historyPrice)
            self.mode = "opener"
            nextOpenerObs = self.featureGenerators[self.mode].getFeatByDatetime(nextDate, self.historyPrice)

            rewardDict = {}
            info["limitOrder"] = False
            if saveMode == "buyer":
                limitTrigger = self.terminal.checkLimitTrigger(self.symbol)
                if limitTrigger:
                    print("Stop/take order triggered.")
                    info["nextOpenerState"] = nextOpenerObs
                    info["limitOrder"] = True
                else:
                    self.terminal.closeOrder(self.order)
                self.order = None
                rewardDict[0] = 0
                rewardDict[1] = (nextHistoryRow[1]["close"] - (self.openPoint[1]["open"] + 0.00018)) * self.lotCoef * self.lotSize
            elif saveMode == "seller":
                limitTrigger = self.terminal.checkLimitTrigger(self.symbol)
                if limitTrigger:
                    print("Stoploss/takeprofit order triggered.")
                    info["nextOpenerState"] = nextOpenerObs
                    info["limitOrder"] = True
                else:
                    self.terminal.closeOrder(self.order)
                self.order = None
                rewardDict[0] = 0
                rewardDict[1] = (self.openPoint[1]["open"] - (nextHistoryRow[1]["close"] + 0.00018)) * self.lotCoef * self.lotSize

            if self.renderFlag == True:
                self.render(self.iStep, nextHistoryRow[1]["open"], nextHistoryRow[1]["close"], action)
            self.deposit += rewardDict[1]
            if self.iStep % 20 == 0:
                print("{}: ".format(self.iStep) + str(action) + " | " + str(rewardDict[1]) + " | " + str(self.deposit))

            return nextOpenerObs, nextCloserObs, rewardDict, self.done, info

    def getNextBar(self, checkTime = 10):
        lastUpdate = self.extractLastDate(self.historyPrice)
        while (True):  # wait for the next update to synchronize real time and update time
            self.updateHistoryPrice()
            freshDateTime = self.extractLastDate(self.historyPrice)
            if freshDateTime != lastUpdate:
                break
            else:
                time.sleep(checkTime)
        nextBar = next( self.historyPrice.tail(1).iterrows() )
        return nextBar

    def extractLastDate(self, df):
        dfDate = parser.parse(list(df.tail(1).index)[0])
        return dfDate

    def updateHistoryPrice(self):
        self.dataUpdater.partialUpdate(self.terminal, self.symbol, self.timeframe)
        self.historyPrice = self.dataManager.getData(self.symbol, self.timeframe)
        self.historyPrice.set_index("datetime", drop=True, inplace=True)
        pass

    def render(self, iStep, open, close, action):
        self.openData.append(open)
        self.closeData.append(close)
        self.xData.append(iStep)
        # plt.scatter(iStep, open, c="black", s=1)
        # plt.scatter(iStep, close, c="blue", s=1)

        color = None
        linewidth = 0.1
        if self.mode == "opener":
            if action == "hold": color = "black"
            elif action == "buy":
                color = "green"
                linewidth = 0.1
            elif action == "sell":
                color = "red"
                linewidth = 0.1
        elif self.mode == "buyer":
            if action == "buy":
                color = "blue"
                linewidth = 0.1
            elif action == "hold": color = "cyan"
        elif self.mode == "seller":
            if action == "sell":
                color = "magenta"
                linewidth = 0.1
            elif action == "hold": color = "yellow"

        # plt.plot(self.xData, self.openData, c="black", linewidth=0.01)
        # plt.plot(self.xData, self.closeData, c="orange", linewidth=0.01)
        plt.plot([iStep, iStep], [open, close], c=color, linewidth=linewidth)
        # plt.scatter( iStep, close, c=color[action], s=1 )
        # plt.pause(0.0000000001)

        # if iStep % 800 == 0:
        #    plt.clf()

        pass

    def setMode(self, mode):
        if mode in ["opener", "buyer", "seller"]:
            self.mode = mode
        else:
            raise ValueError("Not correct mode. Use only: \"opener\", \"buyer\", \"seller\"")
        pass