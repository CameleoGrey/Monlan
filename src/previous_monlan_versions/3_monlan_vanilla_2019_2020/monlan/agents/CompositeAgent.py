import numpy as np
from copy import deepcopy
from datetime import datetime

class CompositeAgent():
    def __init__(self, opener, holdBuyer, holdSeller):
        self.agents = { "opener": opener,
                        "buy": holdBuyer,
                        "sell": holdSeller }
        self.openerActionsDict = {0: "buy", 1: "hold", 2: "sell"}
        self.buyerActionDict = {0: "hold", 1: "buy"}
        self.sellerActionDict = {0: "hold", 1: "sell"}
        self.nSteps = { "opener": 0, "buy": 0, "sell": 0 }
        self.activeAgent = "opener"

        self.maxDeposit = -10000000
        self.maxBackTestReward = -10000000
        self.lastSaveEp = -1

        pass

    # do opener action
    # if action == hold add experience to opener
    # else save state of open state and switch active agent to closer
    # while not close add experience to closer
    # if close position then add experience to closer, switch to opener
    # add saved obs, action and reward of closed position to opener too
    # repeat opener cycle
    def fit_agent(self, env, nEpisodes, plotScores, saveBest = True, saveFreq=5, nWarmUp=0,
                  uniformEps = False, synEps = True, backTestEnv=None, continueFit=False,
                  saveDir = None, saveName = None):
        scores, episodes = [], []

        if continueFit == False:
            self.maxDeposit = -10000000
            self.maxBackTestReward = -10000000
            self.lastSaveEp = -1

        uniEpsList = []
        if uniformEps:
            for i in range(nWarmUp):
                uniEpsList.append(1.0)
            epsStep = 1.0 / nEpisodes
            for i in range(nEpisodes):
                uniEpsList.append( 1.0 - i * epsStep )
            nEpisodes = nEpisodes + nWarmUp

        for e in range(nEpisodes):
            done = False
            score = 0

            #for key in self.nSteps.keys():
            #    self.nSteps[key] = 0

            openState = env.reset()

            while not done:
                # get action for the current state and go one step in environment
                openAction = self.agents["opener"].get_action(openState)

                ##################################################
                if env.parallelOpenState == "closed_original_deal":
                    if env.savedParallelAction == "buy":
                        openAction = 2
                        self.openerActionsDict[openAction] = "sell"
                    elif env.savedParallelAction == "sell":
                        openAction = 0
                        self.openerActionsDict[openAction] = "buy"
                ##################################################

                next_state, openReward, openDone, openInfo = env.step(openAction)
                if openDone:
                    done = True

                elif self.openerActionsDict[openAction] in ["buy", "sell"]:
                    chosenCloser = self.agents[ self.openerActionsDict[openAction] ]
                    closerState = deepcopy(next_state)
                    closerName = self.openerActionsDict[openAction]
                    while True: # buy/sell for closer
                        closerAction = chosenCloser.get_action(closerState)
                        if closerAction == 1:
                            nextOpenerState, nextCloserState, closerReward, closerDone, closerInfo = env.step(closerAction)
                            if closerDone == True:
                                done = True
                                break

                            chosenCloser.append_sample(closerState, 1, closerReward[1], nextCloserState, closerDone)
                            #chosenCloser.append_sample(closerState, 0, closerReward[0], nextCloserState, closerDone)
                            chosenCloser.train_model()
                            self.nSteps[closerName] += 1
                            if self.nSteps[closerName] % chosenCloser.batch_size == 0:
                                self.nSteps[closerName] = 0
                                chosenCloser.update_target_model()
                                ######################################
                                if uniformEps:
                                    for agentName in self.agents.keys():
                                        self.agents[agentName].epsilon = uniEpsList[e]
                                if e < nWarmUp:
                                    for agentName in self.agents.keys():
                                        self.agents[agentName].epsilon = 1.0
                                if synEps:
                                    for agentName in self.agents.keys():
                                        self.agents[agentName].epsilon = self.agents["opener"].epsilon
                                ######################################
                            if closerDone:
                                # every episode update the target model to be same with model
                                chosenCloser.update_target_model()
                                print("{}: ".format(env.iStep) + str(env.deposit))

                            openReward = closerReward[1]
                            openDone = closerDone
                            self.agents["opener"].append_sample(openState, openAction, openReward, nextOpenerState, openDone)
                            #self.agents["opener"].append_sample(openState, openAction, openReward, openState ,openDone)
                            openState = nextOpenerState
                            self.agents["opener"].train_model()
                            self.printInfo(env)
                            break
                        elif closerAction == 0:
                            next_state, closerReward, closerDone, closerInfo = env.step(closerAction)
                            if closerDone:
                                done = True
                                chosenCloser.update_target_model()
                                # if hold up to end of history data then evaluate action like close position
                                #closerAction = 1
                                break

                            #chosenCloser.append_sample(closerState, 1, closerReward[1], next_state, closerDone)
                            chosenCloser.append_sample(closerState, 0, closerReward[0], next_state, closerDone)
                            closerState = next_state
                            ###########################################
                            chosenCloser.train_model()
                            if self.nSteps[closerName] % chosenCloser.batch_size == 0:
                                self.nSteps[closerName] = 0
                                chosenCloser.update_target_model()
                            ###########################################
                                ######################################
                                if uniformEps:
                                    for agentName in self.agents.keys():
                                        self.agents[agentName].epsilon = uniEpsList[e]
                                if e < nWarmUp:
                                    for agentName in self.agents.keys():
                                        self.agents[agentName].epsilon = 1.0
                                if synEps:
                                    for agentName in self.agents.keys():
                                        self.agents[agentName].epsilon = self.agents["opener"].epsilon
                                ######################################
                            if closerInfo["limitOrder"] == True:
                                openState = closerInfo["nextOpenerState"]
                                openReward = closerReward[0]
                                self.printInfo(env)
                                break
                            self.printInfo(env)
                else:
                    if openDone:
                        done = True

                    self.agents["opener"].append_sample(openState, openAction, openReward, next_state, openDone)
                    #self.agents["opener"].append_sample(openState, openAction, openReward, openState, openDone)
                    openState = next_state
                    self.agents["opener"].train_model()
                    self.printInfo(env)

                score += openReward
                self.nSteps["opener"] += 1
                if self.nSteps["opener"] % self.agents["opener"].batch_size == 0: #memory_size?
                    self.nSteps["opener"] = 0
                    self.agents["opener"].update_target_model()
                    ######################################
                    if uniformEps:
                        for agentName in self.agents.keys():
                            self.agents[agentName].epsilon = uniEpsList[e]
                    if e < nWarmUp:
                        for agentName in self.agents.keys():
                            self.agents[agentName].epsilon = 1.0
                    if synEps:
                        for agentName in self.agents.keys():
                            self.agents[agentName].epsilon = self.agents["opener"].epsilon

                if done:
                    # every episode update the target model to be same with model
                    self.agents["opener"].update_target_model()
                    print("{}: ".format(env.iStep) + str(env.deposit))
                    ######################################
                    if uniformEps:
                        self.agents["opener"].epsilon = uniEpsList[e]
                    if e < nWarmUp:
                        self.agents["opener"].epsilon = 1.0
                    if synEps:
                        for agentName in self.agents.keys():
                            self.agents[agentName].epsilon = self.agents["opener"].epsilon
                    ######################################

                    # every episode, plot the play time
                    if plotScores == True:
                        import matplotlib.pyplot as plt
                        scores.append(score)
                        episodes.append(e)
                        plt.close()
                        plt.plot(episodes, scores, 'b')
                        if saveDir is None:
                            plt.savefig("./test_dqn.png")
                        else:
                            plt.savefig("{}score_plot.png".format(saveDir))
                    print("episode:", e, "  score:", score, "  memory length:",
                          len(self.agents["opener"].memory), "  epsilon:", self.agents["opener"].epsilon)

            # save the model
            if saveBest and env.deposit > self.maxDeposit and backTestEnv is None:
                self.maxDeposit = env.deposit
                print("Save new best model. Deposit: {}".format(self.maxDeposit))
                if saveDir is None and saveName is None:
                    self.save_agent("./", "best_composite")
                else:
                    self.save_agent(saveDir, saveName)
                self.lastSaveEp = e
            elif saveBest and backTestEnv is not None:
                backStat = self.use_agent(backTestEnv)
                backReward = np.sum(backStat)
                avgReward = backReward / len(backStat)

                if self.maxBackTestReward < backReward:
                    self.maxBackTestReward = backReward
                #if self.maxBackTestReward < avgReward:
                #    self.maxBackTestReward = avgReward
                    print("Save new best model. Backtest reward: {} | Avg reward: {}".format(backReward, avgReward))
                    if saveDir is None and saveName is None:
                        self.save_agent("./", "best_composite")
                        print("Best agent saved")
                    else:
                        self.save_agent(saveDir, saveName)
                    self.lastSaveEp = e

            elif saveBest == False and e % saveFreq == 0:
                if saveDir is None and saveName is None:
                    self.save_agent("./", "checkpoint_composite")
                else:
                    self.save_agent(saveDir, saveName)
                self.lastSaveEp = e
        return self.lastSaveEp

    def use_agent(self, env, timeConstraint=None):

        startUseTime = datetime.now()
        savedEps = {}
        for agentName in self.agents.keys():
            savedEps[agentName] = self.agents[agentName].epsilon

        #####################################
        #self.agents["opener"].epsilon = 0.0
        #self.agents["buy"].epsilon = 0.125
        #self.agents["sell"].epsilon = 0.125
        ######################################

        #######################################
        for agentName in self.agents.keys():
            self.agents[agentName].epsilon = 0.0
        for agentName in self.agents.keys():
            print( agentName + "_eps: " + str(self.agents[agentName].epsilon))
        ########################################

        dealsStatistics=[]
        score = 0
        openState = env.reset()
        done = False

        while not done:
            # get action for the current state and go one step in environment
            openAction = self.agents["opener"].get_action(openState)
            next_state, openReward, openDone, openInfo = env.step(openAction)
            if openDone:
                done = True
            elif self.openerActionsDict[openAction] in ["buy", "sell"]:
                chosenCloser = self.agents[self.openerActionsDict[openAction]]
                closerState = deepcopy(next_state)
                closerName = self.openerActionsDict[openAction]
                while True:  # buy/sell for closer
                    closerAction = chosenCloser.get_action(closerState)
                    if closerAction == 1:
                        nextOpenerState, nextCloserState, closerReward, closerDone, closerInfo = env.step(
                            closerAction)
                        if closerDone == True:
                            done = True
                            break

                        self.nSteps[closerName] += 1
                        if closerDone:
                            print("{}: ".format(env.iStep) + str(env.deposit))

                        openReward = closerReward[1]
                        openState = nextOpenerState
                        dealsStatistics.append(closerReward[1])
                        self.printInfo(env)

                        if timeConstraint is not None:
                            currentUseTime = datetime.now() - startUseTime
                            if currentUseTime > timeConstraint:
                                done = True
                                break
                        break

                    elif closerAction == 0:
                        next_state, closerReward, closerDone, closerInfo = env.step(closerAction)
                        if closerDone:
                            done = True
                            break
                        if closerInfo["limitOrder"] == True:
                            openState = closerInfo["nextOpenerState"]
                            openReward = closerReward[0]
                            dealsStatistics.append(closerReward[0])
                            break
                        """if timeConstraint is not None:
                            currentUseTime = datetime.now() - startUseTime
                            if currentUseTime > timeConstraint:
                                closePos = 1
                                env.step(closePos)
                                done = True
                                break"""
                        closerState = next_state
                        self.printInfo(env)
            else:
                if openDone:
                    done = True

                if timeConstraint is not None:
                    currentUseTime = datetime.now() - startUseTime
                    if currentUseTime > timeConstraint:
                        done = True
                        break

                openState = next_state
                self.printInfo(env)
            score += openReward
            self.nSteps["opener"] += 1

            if done:
                print("{}: ".format(env.iStep) + str(env.deposit))

        for agentName in self.agents.keys():
            self.agents[agentName].epsilon = savedEps[agentName]
        return dealsStatistics

    def printInfo(self, env):
        if env.iStep % 100 == 0:
            print("Step: {}".format(env.iStep) + " | Deposit: {}".format(str(env.deposit)))
            ######################################
            print("oe: {}".format(self.agents["opener"].epsilon), end=" ")
            print("be: {}".format(self.agents["buy"].epsilon), end=" ")
            print("se: {}".format(self.agents["sell"].epsilon))
            #######################################
            print("om: {}".format(len(self.agents["opener"].memory)), end=" ")
            print("bm: {}".format(len(self.agents["buy"].memory)), end=" ")
            print("sm: {}".format(len(self.agents["sell"].memory)))
            #######################################

    def save_agent(self, path, name):
        #import joblib
        #with open(path + "/" + name + ".pkl", mode="wb") as agentFile:
        #    joblib.dump(self, agentFile)

        for agentName in self.agents.keys():
            self.agents[agentName].save_agent( path, name + "_" + agentName )

        pass

    def load_agent(self, path, name, dropSupportModel = False):
        #import joblib
        #loadedAgent = None
        #with open(path + "/" + name + ".pkl", mode="rb") as agentFile:
        #    loadedAgent = joblib.load(agentFile)

        for agentName in self.agents.keys():
            self.agents[agentName] = self.agents[agentName].load_agent( path, name + "_" + agentName, dropSupportModel=dropSupportModel )
        self.nSteps = {"opener": 0, "buy": 0, "sell": 0}
        self.activeAgent = "opener"

        return self