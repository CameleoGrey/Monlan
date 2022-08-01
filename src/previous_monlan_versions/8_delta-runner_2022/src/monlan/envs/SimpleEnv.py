

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers

class SimpleEnv():

    class SimpleActionSpace():
        def __init__(self):
            self.actionsDict = { 0: "buy", 1: "hold", 2: "sell" }
            self.n = 3
            pass

    def __init__(self, historyPrice, obsFeatList = ["open"], renderFlag=True):
        self.historyPrice = historyPrice
        self.historyPrice.set_index("datetime", drop=True, inplace=True)
        self.iter = None
        self.action_space = self.SimpleActionSpace()
        self.observation_space = np.array( obsFeatList )
        self.done = False
        self.deposit = 300.0
        self.lotSize = 15.0
        self.renderFlag = renderFlag

        self.openData = []
        self.closeData = []
        self.xData = []
        self.iStep = 0

        pass

    def reset(self):
        self.done = False
        self.deposit = 300.0
        self.openData = []
        self.closeData = []
        self.xData = []
        self.iStep = 0
        self.iter = self.historyPrice.iterrows()
        if self.renderFlag == True:
            plt.close()

        randPos = np.random.randint(1, 2, 1)[0]
        obs = None
        for i in range( randPos ):
            obs = next(self.iter)

        selectedList = []
        for feat in self.observation_space:
            selectedList.append( obs[1][feat] )
        obs = np.array( selectedList )

        return obs

    def seed(self, seed=None):
        np.random.seed(seed)
        pass

    def step(self, action):
        action = self.action_space.actionsDict[action]
        obs = next( self.iter, None )
        info = {}
        reward = None

        # check end of
        if obs == None:
            self.done = True
            reward = 0
            selectedList = []
            for feat in self.observation_space:
                selectedList.append(0.0)
            obs = np.array(selectedList)
            if self.renderFlag == True:
                plt.savefig("./test_plot.png", dpi=2000)
            return obs, reward, self.done, info

        if action == "buy":
            reward = (obs[1]["close"] - obs[1]["open"])
        elif action == "hold":
            reward = 0
        elif action == "sell":
            reward = (obs[1]["open"] - obs[1]["close"])
        reward = reward * self.lotSize
        self.deposit += reward
        if self.deposit <= 0.0:
            self.done = True

        if self.renderFlag == True:
            self.render(self.iStep, obs[1]["open"], obs[1]["close"], action)

        selectedList = []
        for feat in self.observation_space:
            selectedList.append(obs[1][feat])
        obs = np.array(selectedList)

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