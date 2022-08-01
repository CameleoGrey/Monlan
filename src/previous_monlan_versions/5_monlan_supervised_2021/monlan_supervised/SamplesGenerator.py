
from tqdm import tqdm
import numpy as np
import gc

class SamplesGenerator():
    def __init__(self, featureGen, lotSize=0.1, lotCoef=100000, spreadCoef = 0.00001):
        self.featureGen = featureGen
        self.lotSize = lotSize
        self.lotCoef = lotCoef
        self.spreadCoef = spreadCoef
        pass

    def generateOpenerSamples(self, historyData, windowSize = 10):

        startDt = self.featureGen.getMinDate(historyData)
        stepCount = historyData.index[-1] - startDt

        x_samples = []
        y_samples = []
        #########################
        # build own index->["open", "close"] map to avoid to use slow pandas.loc[index]
        indexArr = historyData.index.values
        featArr = historyData[["open", "close", "spread"]].values
        indexFeatsMap = {}
        for i in range(len(indexArr)):
            indexFeatsMap[indexArr[i]] = featArr[i]
        #########################
        for i in tqdm(range(startDt, startDt + stepCount - windowSize - 1), desc="Generating opener samples", colour="green"):
            maxBuyerReward = -4e+10
            maxSellerReward = -4e+10
            bestBuyHold = 0
            bestSellHold = 0
            cummHold = 0
            for j in range(i+1, i+1+windowSize):
                addCoef = 1.0 / (1.0 * windowSize)
                cummHold += addCoef * (abs(indexFeatsMap[j][1] - indexFeatsMap[j][0]) + self.spreadCoef * indexFeatsMap[j][2]) * self.lotCoef * self.lotSize

                localBuyerReward = (indexFeatsMap[j][1] - (indexFeatsMap[i][0] + self.spreadCoef * indexFeatsMap[i][2])) * self.lotCoef * self.lotSize
                if localBuyerReward > maxBuyerReward:
                    maxBuyerReward = localBuyerReward
                    bestBuyHold = cummHold

                localSellerReward = (indexFeatsMap[i][0] - (indexFeatsMap[j][1] + self.spreadCoef * indexFeatsMap[j][2])) * self.lotCoef * self.lotSize
                if localSellerReward > maxSellerReward:
                    maxSellerReward = localSellerReward
                    bestSellHold = cummHold

            x_sample = self.featureGen.getManyPointsFeat(i, historyData)
            y_sample = [maxBuyerReward, (bestBuyHold + bestSellHold) / 2, maxSellerReward]
            x_samples.append(x_sample)
            y_samples.append(y_sample)
            # gc.collect()

        gc.collect()
        x_samples = np.array( x_samples, dtype=np.float32)
        y_samples = np.array( y_samples )

        return x_samples, y_samples

    def generateBuyerSamples(self, historyData, windowSize = 10):

        startDt = self.featureGen.getMinDate(historyData)
        stepCount = historyData.index[-1] - startDt

        x_samples = []
        y_samples = []
        usedRewardIds = set()
        usedLossIds = set()
        #########################
        # build own index->["open", "close"] map to avoid to use slow pandas.loc[index]
        indexArr = historyData.index.values
        featArr = historyData[["open", "close", "spread"]].values
        indexFeatsMap = {}
        for i in range(len(indexArr)):
            indexFeatsMap[indexArr[i]] = featArr[i]
        #########################
        for i in tqdm(range(startDt, startDt + stepCount - windowSize - 1), desc="Generating buyer samples", colour="green"):
            maxBuyerReward = -4e+10
            maxBuyerRewardInd = None
            minBuyerReward = 4e+10
            minBuyerRewardInd = None
            savedRewardHold = None
            savedLossHold = None
            meanHoldReward = 0
            for j in range(i+1, i+1+windowSize):
                mhCoef = 1.0 / (1.0 * windowSize)
                meanHoldReward += mhCoef * (abs(indexFeatsMap[j][1] - indexFeatsMap[j][0]) + self.spreadCoef * indexFeatsMap[j][2]) * self.lotCoef * self.lotSize
                #sumHoldLoss = (indexFeatsMap[j+1][1] - (indexFeatsMap[i][0] + self.spreadCoef * indexFeatsMap[i][2])) * self.lotCoef * self.lotSize

                localBuyerReward = (indexFeatsMap[j][1] - (indexFeatsMap[i][0] + self.spreadCoef * indexFeatsMap[i][2])) * self.lotCoef * self.lotSize
                if localBuyerReward > maxBuyerReward:
                    maxBuyerReward = localBuyerReward
                    maxBuyerRewardInd = j
                    savedRewardHold = meanHoldReward #estimation of mean loss before achieving best deal point
                if localBuyerReward < minBuyerReward:
                    minBuyerReward = localBuyerReward
                    minBuyerRewardInd = j
                    savedLossHold = -meanHoldReward #estimation of mean reward before achieving worst deal point

            # one have best and worst case. As a good trader one needs to learn middle path
            # reinforce waiting by the best and worst predicting rewards
            # when one achieves point where predicted waiting reward
            """x_sample = self.featureGen.getManyPointsFeat(i, historyData)
            y_best = [maxBuyerReward, savedRewardHold]
            y_worst = [minBuyerReward, savedLossHold]
            x_samples.append(x_sample)
            y_samples.append(y_best)
            x_samples.append(x_sample)
            y_samples.append(y_worst)
            # gc.collect()"""

            x_sample = self.featureGen.getManyPointsFeat(i, historyData)
            y_mean = [(maxBuyerReward + minBuyerReward) / 2.0, (savedRewardHold + savedLossHold) / 2]
            x_samples.append(x_sample)
            y_samples.append(y_mean)
            # gc.collect()

            # x_sample = self.featureGen.getManyPointsFeat(maxBuyerRewardInd, historyData)
            """if maxBuyerRewardInd not in usedRewardIds:
                usedRewardIds.add(maxBuyerRewardInd)
                x_sample = self.featureGen.getManyPointsFeat(maxBuyerRewardInd, historyData)

                # if one predicts good deal reinforce waiting, especially if one predicts best deal in the future
                if maxBuyerRewardInd > minBuyerRewardInd:
                    y_sample = [maxBuyerReward, minBuyerReward]
                else:
                    y_sample = [maxBuyerReward, savedRewardHold]
                x_samples.append(x_sample)
                y_samples.append(y_sample)
                #gc.collect()

            if minBuyerRewardInd not in usedLossIds:
                usedLossIds.add(minBuyerRewardInd)
                x_sample = self.featureGen.getManyPointsFeat(minBuyerRewardInd, historyData)

                # if one predicts worst deal reinforce closing, especially if one predicts best deal in the future
                if maxBuyerRewardInd > minBuyerRewardInd:
                    y_sample = [minBuyerReward, maxBuyerReward]
                else:
                    y_sample = [minBuyerReward, -savedLossHold]
                x_samples.append(x_sample)
                y_samples.append(y_sample)
                #gc.collect()"""

        gc.collect()
        x_samples = np.array( x_samples, dtype=np.float32 )
        y_samples = np.array( y_samples )

        return x_samples, y_samples

    def generateSellerSamples(self, historyData, windowSize = 10):

        startDt = self.featureGen.getMinDate(historyData)
        stepCount = historyData.index[-1] - startDt

        x_samples = []
        y_samples = []
        usedRewardIds = set()
        usedLossIds = set()

        #########################
        #build own index->["open", "close"] map to avoid to use slow pandas.loc[index]
        indexArr = historyData.index.values
        featArr = historyData[["open", "close", "spread"]].values
        indexFeatsMap = {}
        for i in range(len(indexArr)):
            indexFeatsMap[indexArr[i]] = featArr[i]
        #########################

        for i in tqdm(range(startDt, startDt + stepCount - windowSize - 1), desc="Generating seller samples", colour="green"):
            maxSellerReward = -4e+10
            maxSellerRewardInd = None
            minSellerRewardInd = None
            minSellerReward = 4e+10
            savedRewardHold = None
            savedLossHold = None
            meanHoldReward = 0
            for j in range(i+1, i+1+windowSize):
                addCoef = 1.0 / (1.0 * windowSize)
                meanHoldReward += addCoef * (abs(indexFeatsMap[j][0] - indexFeatsMap[j][1]) + self.spreadCoef * indexFeatsMap[j][2]) * self.lotCoef * self.lotSize
                #meanHoldReward = (indexFeatsMap[i][0] - (indexFeatsMap[j+1][1] + self.spreadCoef * indexFeatsMap[j+1][2])) * self.lotCoef * self.lotSize

                localSellerReward = (indexFeatsMap[i][0] - (indexFeatsMap[j][1] + self.spreadCoef * indexFeatsMap[j][2])) * self.lotCoef * self.lotSize
                if localSellerReward > maxSellerReward:
                    maxSellerReward = localSellerReward
                    maxSellerRewardInd = j
                    savedRewardHold = meanHoldReward #estimation of mean reward before achieving best deal point
                if localSellerReward < minSellerReward:
                    minSellerReward = localSellerReward
                    minSellerRewardInd = j
                    savedLossHold = -meanHoldReward #estimation of mean loss before achieving worst deal point

            # one have best and worst case. As a good trader one needs to learn middle path
            # reinforce waiting by the best and worst predicting rewards
            # when one achieves point where predicted waiting reward
            """x_sample = self.featureGen.getManyPointsFeat(i, historyData)
            y_best = [maxSellerReward, savedRewardHold]
            y_worst = [minSellerReward, savedLossHold]
            x_samples.append(x_sample)
            y_samples.append(y_best)
            x_samples.append(x_sample)
            y_samples.append(y_worst)
            # gc.collect()"""

            x_sample = self.featureGen.getManyPointsFeat(i, historyData)
            y_mean = [(maxSellerReward + minSellerReward) / 2.0, (savedRewardHold + savedLossHold) / 2]
            x_samples.append(x_sample)
            y_samples.append(y_mean)
            # gc.collect()

            # x_sample = self.featureGen.getManyPointsFeat(maxBuyerRewardInd, historyData)
            """if maxSellerRewardInd not in usedRewardIds:
                usedRewardIds.add(maxSellerRewardInd)
                x_sample = self.featureGen.getManyPointsFeat(maxSellerRewardInd, historyData)
                # if one predicts good deal reinforce waiting, especially if one predicts worst deal in the future
                if maxSellerRewardInd > minSellerRewardInd:
                    y_sample = [maxSellerReward, minSellerReward]
                else:
                    y_sample = [maxSellerReward, -savedLossHold]
                x_samples.append(x_sample)
                y_samples.append(y_sample)
                #gc.collect()

            if minSellerRewardInd not in usedLossIds:
                usedLossIds.add(minSellerRewardInd)
                x_sample = self.featureGen.getManyPointsFeat(minSellerRewardInd, historyData)
                # if one predicts worst deal reinforce closing, especially if one predicts best deal in the future
                if maxSellerRewardInd > minSellerRewardInd:
                    y_sample = [minSellerReward, maxSellerReward]
                else:
                    y_sample = [minSellerReward, -savedLossHold]
                x_samples.append(x_sample)
                y_samples.append(y_sample)
                #gc.collect()"""

        gc.collect()
        x_samples = np.array( x_samples, dtype=np.float32 )
        y_samples = np.array( y_samples )

        return x_samples, y_samples