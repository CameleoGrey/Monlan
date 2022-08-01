import numpy as np
import time
from gym import Env
from dateutil import parser

class RealEnv(Env):

    class SimpleActionSpace():
        def __init__(self):
            self.actionsDict = { 0: "buy", 1: "hold", 2: "sell" }
            self.n = 3
            pass

    def __init__(self, symbol, timeframe, terminal, dataUpdater, dataManager, featureFactory, obsFeatList = ["open"]):
        self.symbol = symbol
        self.timeframe = timeframe
        self.terminal = terminal
        self.dataUpdater = dataUpdater
        self.dataManager = dataManager
        self.featureFactory = featureFactory
        self.action_space = self.SimpleActionSpace()
        self.observation_space = np.array( obsFeatList )
        self.done = False
        self.deposit = 300.0
        self.lotSize = 0.01
        self.lastUpdate = None

        pass

    def reset(self):
        self.done = False
        self.deposit = 300.0

        self.dataUpdater.partialUpdate(self.terminal, self.symbol, self.timeframe)
        rawData = self.dataManager.getData(self.symbol, self.timeframe)
        rawData = self.featureFactory.extractFeature( rawData, self.observation_space )

        lastRow = rawData.tail(1)
        self.lastUpdate = parser.parse(lastRow["datetime"].values[0])
        while (True): #wait for the next update to synchronize real time and update time
            self.dataUpdater.partialUpdate(self.terminal, self.symbol, self.timeframe)
            rawData = self.dataManager.getData(self.symbol, self.timeframe)
            rawData = self.featureFactory.extractFeature(rawData, self.observation_space)
            lastRow = rawData.tail(1)
            freshDateTime = parser.parse(lastRow["datetime"].values[0])
            if freshDateTime != self.lastUpdate:
                self.lastUpdate = freshDateTime
                break
            else:
                time.sleep(10) #TODO: add custom sleep time

        lastRow = next(lastRow.iterrows())
        selectedList = []
        for feat in self.observation_space:
            selectedList.append( lastRow[1][feat] )
        obs = np.array( selectedList )

        return obs

    def step(self, action):
        action = self.action_space.actionsDict[action]
        info = {}
        reward = None

        order = None
        if action in ["buy", "sell"]:
            order = self.terminal.placeOrder(action, self.symbol, self.lotSize)

        lastRow = None
        while (True):
            self.dataUpdater.partialUpdate(self.terminal, self.symbol, self.timeframe)
            rawData = self.dataManager.getData(self.symbol, self.timeframe)
            rawData = self.featureFactory.extractFeature(rawData, self.observation_space)
            lastRow = rawData.tail(1)
            freshDateTime = parser.parse(lastRow["datetime"].values[0])
            if freshDateTime != self.lastUpdate:
                self.lastUpdate = freshDateTime
                break
            else:
                time.sleep(10) #TODO: add custom sleep time
        if action in ["buy", "sell"]:
            self.terminal.closeOrder(order)

        obs = next(lastRow.iterrows())

        if action == "buy":
            reward = (obs[1]["close"] - obs[1]["open"])
        elif action == "hold":
            reward = 0
        elif action == "sell":
            reward = (obs[1]["open"] - obs[1]["close"])
        reward = reward * self.lotSize * 100000 #TODO: add one lote size based on the deposit type  (ECN, Cent, Std)
        self.deposit += reward
        if self.deposit <= 0.0:
            self.done = True

        selectedList = []
        for feat in self.observation_space:
            selectedList.append(obs[1][feat])
        obs = np.array(selectedList)

        return obs, reward, self.done, info
