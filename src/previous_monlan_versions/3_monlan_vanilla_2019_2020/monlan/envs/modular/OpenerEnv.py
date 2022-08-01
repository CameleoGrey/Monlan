
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
from dateutil import parser
from monlan.agents.RiskManager import RiskManager
import time

class OpenerEnv():
    class OpenerActionSpace():
        def __init__(self):
            self.actionsDict = { 0: "buy", 1: "hold", 2: "sell" }
            self.n = 3
            pass

    def __init__(self, historyPrice, openerFeatureGen, buyerFeatureGen, sellerFeatureGen, startDeposit = 300,
                 lotSize=0.1, lotCoef=100000, spread = 18, spreadCoef = 0.00001, stopType = "const", takeType = "const",
                 stoplossPuncts=100, takeprofitPuncts=500, stopPos = 0, takePos = 0, maxLoss = 40000, maxTake = 40000,
                 riskPoints=110, riskLevels=5, renderFlag=True, renderDir=None, renderName=None ):
        self.historyPrice = historyPrice.copy()
        self.historyPrice.set_index("datetime", drop=True, inplace=True)
        self.historyIter = None
        self.done = False
        self.startDeposit = startDeposit
        self.deposit = self.startDeposit
        self.lotSize = lotSize
        self.lotCoef = lotCoef
        self.spread = spread
        self.spreadCoef = spreadCoef
        self.renderFlag = renderFlag
        self.renderDir = renderDir
        self.renderName = renderName
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
                                       spreadCoef=self.spreadCoef)

        self.openData = []
        self.closeData = []
        self.xData = []
        self.iStep = 0

        self.actorNames = ["opener"]
        self.featureGenerators = {"opener": openerFeatureGen}
        self.action_space = {"opener": self.OpenerActionSpace()}
        self.observation_space = {"opener": openerFeatureGen.featureShape}
        self.openPoint = None
        self.mode = "opener"
        self.sumLoss = 0

        pass

    def reset(self):
        self.openPoint = None
        self.sumLoss = 0
        self.done = False
        self.deposit = self.startDeposit
        self.openData = []
        self.closeData = []
        self.xData = []
        self.iStep = 0
        if self.renderFlag == True:
            plt.close()

        self.historyIter = self.historyPrice.iterrows()
        startDate = self.getStartDate()
        self.holdIter = self.historyPrice.iterrows()
        holdDate = next(self.holdIter)
        while holdDate[0] != startDate:
            holdDate = next( self.holdIter, None )
        holdDate = next(self.holdIter, None)[0]

        obs = self.featureGenerators[self.mode].getFeatByDatetime(startDate, self.historyPrice)
        return obs

    def getStartDate(self):
        minDate = self.featureGenerators["opener"].getMinDate(self.historyPrice)
        startDate = next(self.historyIter)[0]
        while startDate != minDate:
            startDate = next(self.historyIter)[0]
        return startDate

    def step(self, action):
        info = {}
        reward = None
        action = self.action_space[self.mode].actionsDict[action]
        self.iStep += 1

        nextHistoryRow = next(self.historyIter, None)
        nextHoldRow = next(self.holdIter, None)
        if nextHistoryRow is None or nextHoldRow is None:
            self.done = True
            reward = 0
            selectedList = []
            for feat in self.featureGenerators[self.mode].featureList:
                selectedList.append(0.0)
            obs = np.array(selectedList)
            return obs, reward, self.done, info

        nextDate = nextHistoryRow[0]
        if action in ["buy", "sell"]:
            self.openPoint = deepcopy(nextHistoryRow)
            self.sumLoss = 0
            reward = 0

        if action == "buy":
            if self.stopType == "adaptive" or self.takeType == "adaptive":
                self.setAdaptiveLimits(nextDate, self.historyPrice, self.openPoint[1]["open"])
            info["state"] = "buyer"
        elif action == "sell":
            if self.stopType == "adaptive" or self.takeType == "adaptive":
                self.setAdaptiveLimits(nextDate, self.historyPrice, self.openPoint[1]["open"])
            info["state"] = "seller"
        elif action == "hold":
            #self.sumLoss += -abs((nextHoldRow[1]["open"] - (nextHistoryRow[1]["open"] + 0.5 * nextHistoryRow[1]["open"])) * self.lotCoef * self.lotSize)
            #self.sumLoss += -abs((nextHistoryRow[1]["close"] - (nextHistoryRow[1]["open"] + self.spreadCoef * self.spread)) * self.lotCoef * self.lotSize) * 0.8
            self.sumLoss += -np.pi * self.spreadCoef / 2.0
            reward = self.sumLoss
            info["state"] = "hold"

        obs = self.featureGenerators[self.mode].getFeatByDatetime(nextDate, self.historyPrice)
        return obs, reward, self.done, info

    def setAdaptiveLimits(self, nextDate, historyPrice, openPoint):
        limitsDict = self.riskManager.getAdaptiveLimits(nextDate, historyPrice, openPoint[1]["open"])
        self.stoplossPuncts = limitsDict["stop"]
        self.takeprofitPuncts = limitsDict["take"]
        if self.stoplossPuncts > self.maxLoss:
            self.stoplossPuncts = self.maxLoss
        if self.takeprofitPuncts > self.maxTake:
            self.takeprofitPuncts = self.maxTake