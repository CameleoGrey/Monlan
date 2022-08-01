import gc
import os
import numpy as np
from copy import deepcopy
from datetime import datetime
from src.monlan.utils.save_load import *

class CompositeAgent():
    def __init__(self, opener, holdBuyer, holdSeller, reward_transformer=None):

        self.openerActionsDict = {0: "buy", 1: "hold", 2: "sell"}
        self.buyerActionDict = {0: "hold", 1: "buy"}
        self.sellerActionDict = {0: "hold", 1: "sell"}
        self.nSteps = { "opener": 0, "buyer": 0, "seller": 0 }

        self.agents = { "opener": opener,
                        "buyer": holdBuyer,
                        "seller": holdSeller }
        self.agent = self.agents["opener"]


        self.start_deposit = 300.0
        self.deposit = self.start_deposit
        self.reward_transformer = reward_transformer


        self.best_score = -np.inf
        pass

    def reset(self):
        self.agent = self.agents["opener"]
        self.deposit = self.start_deposit
        pass

    def set_agent(self, agent_name):
        self.agent = self.agents[agent_name]
        pass

    def get_agent_name(self):
        return self.agent.get_name()


    def fit_agent(self, env, nEpisodes, plotScores, saveBest = True, saveFreq=5, nWarmUp=0,
                  uniformEps = False, synEps = True, backTestEnv=None, continueFit=False,
                  saveDir = None, saveName = None):

        for e in range(nEpisodes):
            done = False
            self.reset()
            current_state = env.reset()["opener"]
            while not done:
                if self.get_agent_name() != env.get_state_name():
                    raise ValueError("Incompatible agent")
                action = self.agent.get_action( current_state.current_obs )
                next_state = env.step( action )

                if next_state[list(next_state.keys())[0]].done:
                    done = True
                    #env.render_episode()
                    continue

                for env_name in next_state.keys():
                    sample_reward = next_state[env_name].reward_dict[ next_state[env_name].action ]
                    if self.reward_transformer is not None:
                        sample_reward = self.reward_transformer.transform( sample_reward )
                    self.agents[env_name].append_sample( next_state[env_name].current_obs,
                                                        next_state[env_name].action,
                                                        sample_reward,
                                                        next_state[env_name].next_obs,
                                                        #next_state[env_name].current_obs, # just for fun
                                                        next_state[env_name].done )
                    self.printInfo(env)

                if len(next_state) == 2:
                    current_state = next_state["opener"]
                    done = next_state["opener"].done
                    self.set_agent( "opener" )

                    # train on each closed deal
                    for env_name in next_state.keys():
                        self.agents[env_name].train_model()
                else:
                    current_state = next_state[ list(next_state.keys())[0] ]
                    done = next_state[ list(next_state.keys())[0] ].done
                    current_state_name = env.get_state_name()
                    self.set_agent( current_state_name )
                    #for env_name in next_state.keys():
                    #    self.agents[env_name].train_model()

            if backTestEnv is not None:
                self.back_test_(backTestEnv, e, saveBest, saveFreq, saveDir, saveName)

        pass

    def back_test_(self, backTestEnv, i_episode, saveBest, saveFreq, saveDir = None, saveName = None):

        back_test_score, deals_statistics = self.use_agent(backTestEnv, render_deals=True)
        print("Back-test score: {}".format( back_test_score ))

        if back_test_score >= self.best_score:
            print("New best score: {} | Previous best: {}".format( back_test_score, self.best_score ))
            self.best_score = back_test_score
            if saveBest:
                self.save_agent( saveDir, saveName )

        if saveFreq is not None and saveBest is False:
            if saveFreq == 0: raise ValueError("Save frequency is 0")
            if i_episode % saveFreq == 0:

                #pytorch
                memories = {}
                for agentName in self.agents.keys():
                    memories[agentName] = self.agents[agentName].memory
                    self.agents[agentName].memory = None

                save( self, os.path.join(saveDir, saveName + "_{}.pkl".format(i_episode)) )

                for agentName in self.agents.keys():
                    self.agents[agentName].memory = memories[agentName]

                #keras
                #self.save_agent(saveDir, saveName + "_{}".format(i_episode))

        pass


    def use_agent(self, env, timeConstraint=None, render_deals=False):

        done = False
        self.reset()
        current_state = env.reset()["opener"]
        score = self.start_deposit
        deals_statistics = []

        saved_eps_dict = {}
        for agent_name in self.agents.keys():
            agent_eps = self.agents[agent_name].epsilon
            saved_eps_dict[agent_name] = agent_eps
            self.agents[agent_name].epsilon = 0.0

        while not done:
            action = self.agent.get_action(current_state.current_obs)
            next_state = env.step(action)
            #self.printInfo(env)

            if next_state[list(next_state.keys())[0]].done:
                done = True
                if render_deals:
                    env.render_episode()
                continue

            if len(next_state) == 2:
                current_state = next_state["opener"]
                done = next_state["opener"].done
                reward = next_state["opener"].reward_dict[ next_state["opener"].action ]
                score += reward
                deals_statistics.append( reward )
                self.set_agent("opener")
            else:
                current_state = next_state[list(next_state.keys())[0]]
                done = next_state[list(next_state.keys())[0]].done
                current_state_name = env.get_state_name()
                self.set_agent(current_state_name)

        for agent_name in self.agents.keys():
            self.agents[agent_name].epsilon = saved_eps_dict[agent_name]

        return score, deals_statistics

    def printInfo(self, env):
        if env.iStep % 300 == 0:
            ######################################
            print("oe: {}".format(self.agents["opener"].epsilon), end=" ")
            print("be: {}".format(self.agents["buyer"].epsilon), end=" ")
            print("se: {}".format(self.agents["seller"].epsilon))
            #######################################
            print("om: {}".format(len(self.agents["opener"].memory)), end=" ")
            print("bm: {}".format(len(self.agents["buyer"].memory)), end=" ")
            print("sm: {}".format(len(self.agents["seller"].memory)))
            #######################################

    def save_agent(self, path, name):

        memories = {}
        for agentName in self.agents.keys():
            memories[agentName] = self.agents[agentName].memory
            self.agents[agentName].memory = None

        for agentName in self.agents.keys():
            self.agents[agentName].save_agent( path, name + "_" + agentName )

        for agentName in self.agents.keys():
            self.agents[agentName].memory = memories[agentName]

        pass

    def load_agent(self, path, name, dropSupportModel = False, compile=True):

        #pytorch
        agent = load( os.path.join( path, name + ".pkl" ) )
        self.agents = agent.agents
        del agent
        gc.collect()

        #keras
        #for agentName in self.agents.keys():
        #    self.agents[agentName] = self.agents[agentName].load_agent( path, name + "_" + agentName,
        #                                                                dropSupportModel=dropSupportModel,
        #                                                                compile=compile)

        self.nSteps = {"opener": 0, "buy": 0, "sell": 0}
        self.agent = "opener"

        return self