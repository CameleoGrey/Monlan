
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers

class FeatureGenEnv():

    class SimpleActionSpace():
        def __init__(self):
            self.actionsDict = { 0: "buy", 1: "hold", 2: "sell" }
            self.n = 3
            pass

    def __init__(self, historyPrice, featureGenerator, startDeposit = 300, spread=18, spreadCoef = 0.00001,
                 lotSize=0.1, lotCoef=100000, renderFlag=True):
        self.historyPrice = historyPrice
        self.historyPrice.set_index("datetime", drop=True, inplace=True)
        self.featureGenerator = featureGenerator
        self.historyIter = None
        self.action_space = self.SimpleActionSpace()
        self.observation_space = featureGenerator.featureShape
        self.done = False
        self.startDeposit = startDeposit
        self.deposit = self.startDeposit
        self.lotSize = lotSize
        self.lotCoef = lotCoef
        self.renderFlag = renderFlag
        self.spread = spread
        self.spreadCoef = spreadCoef

        self.openData = []
        self.closeData = []
        self.xData = []
        self.iStep = 0

        pass

    def reset(self):
        self.done = False
        self.deposit = self.startDeposit
        self.openData = []
        self.closeData = []
        self.xData = []
        self.iStep = 0
        self.historyIter = self.historyPrice.iterrows()
        if self.renderFlag == True:
            plt.close()

        nextDate = next(self.historyIter)[0]
        obs = self.featureGenerator.getFeatByDatetime(nextDate, self.historyPrice)

        return obs

    def step(self, action):
        action = self.action_space.actionsDict[action]
        nextHistoryRow = next( self.historyIter, None)
        if nextHistoryRow is None:
            obs = None
        else:
            nextDate = nextHistoryRow[0]
            obs = self.featureGenerator.getFeatByDatetime(nextDate, self.historyPrice)
        info = {}
        reward = None

        # check end of
        if obs is None:
            self.done = True
            reward = 0
            selectedList = []
            for feat in self.featureGenerator.featureList:
                selectedList.append(0.0)
            obs = np.array(selectedList)
            if self.renderFlag == True:
                plt.savefig("./test_plot.png", dpi=2000)
            return obs, reward, self.done, info

        if action == "buy":
            reward = (nextHistoryRow[1]["close"] - nextHistoryRow[1]["open"]) * self.lotCoef * self.lotSize
            #reward = (nextHistoryRow[1]["close"] - (nextHistoryRow[1]["open"] + self.spreadCoef * self.spread)) * self.lotCoef * self.lotSize
        elif action == "hold":
            reward = 0.5 * (nextHistoryRow[1]["open"] - nextHistoryRow[1]["close"]) * self.lotCoef * self.lotSize
            #reward = -0.5 * abs((nextHistoryRow[1]["close"] - (nextHistoryRow[1]["open"] + self.spreadCoef * self.spread))) * self.lotCoef * self.lotSize
        elif action == "sell":
            reward = (nextHistoryRow[1]["open"] - nextHistoryRow[1]["close"]) * self.lotCoef * self.lotSize
            #reward = (nextHistoryRow[1]["open"] - (nextHistoryRow[1]["close"] + self.spreadCoef * self.spread)) * self.lotCoef * self.lotSize
        reward = reward
        self.deposit += reward
        #if self.deposit <= 0.0:
        #    self.done = True

        if self.renderFlag == True:
            self.render(self.iStep, nextHistoryRow[1]["open"], nextHistoryRow[1]["close"], action)

        self.iStep += 1
        if self.iStep % 100 == 0:
            print( "{}: ".format(self.iStep) +  str(action) + " | " + str(reward) + " | " + str(self.deposit) )

        if self.done == True:
            plt.savefig("./test_plot.png", dpi=2000)

        return obs, reward, self.done, info

    def render(self, iStep, open, close, action):
        self.openData.append(open)
        self.closeData.append(close)
        self.xData.append(iStep)
        #plt.scatter(iStep, open, c="black", s=1)
        #plt.scatter(iStep, close, c="blue", s=1)

        color = {}
        color["buy"] = "green"
        color["hold"] = "black"
        color["sell"] = "red"

        #plt.plot(self.xData, self.openData, c="black", linewidth=0.01)
        #plt.plot(self.xData, self.closeData, c="orange", linewidth=0.01)
        plt.plot( [iStep, iStep], [open, close], c=color[action], linewidth=0.1 )
        #plt.scatter( iStep, close, c=color[action], s=1 )
        #plt.pause(0.0000000001)

        #if iStep % 800 == 0:
        #    plt.clf()

        pass