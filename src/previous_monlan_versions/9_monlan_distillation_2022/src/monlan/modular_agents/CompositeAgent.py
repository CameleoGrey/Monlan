import gc
import os
import numpy as np
from copy import deepcopy
from datetime import datetime
from src.monlan.utils.save_load import *

class CompositeAgent():
    def __init__(self, opener, hold_buyer, hold_seller, start_deposit=1000):

        self.opener_actions_dict = {0: "buy", 1: "hold", 2: "sell"}
        self.buyer_action_dict = {0: "hold", 1: "buy"}
        self.seller_action_dict = {0: "hold", 1: "sell"}
        self.n_steps = { "opener": 0, "buyer": 0, "seller": 0 }

        self.agents = { "opener": opener,
                        "buyer": hold_buyer,
                        "seller": hold_seller }
        self.agent = self.agents["opener"]


        self.start_deposit = start_deposit
        self.deposit = self.start_deposit


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


    def fit_agent(self, env, n_episodes, plot_scores, save_best = True, save_freq=5, n_warm_up=0,
                  uniform_eps = False, syn_eps = True, back_test_env=None, continue_fit=False,
                  save_dir = None, save_name = None, excluded_from_training_agents=None):

        for e in range(n_episodes):
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
                    self.agents[env_name].append_sample( next_state[env_name].current_obs,
                                                        next_state[env_name].action,
                                                        sample_reward,
                                                        next_state[env_name].next_obs,
                                                        next_state[env_name].done )
                    self.print_info(env)

                if len(next_state) == 2:
                    current_state = next_state["opener"]
                    done = next_state["opener"].done
                    self.set_agent( "opener" )

                    # train on each closed deal
                    for env_name in next_state.keys():
                        if excluded_from_training_agents is not None:
                            if env_name not in excluded_from_training_agents:
                                self.agents[env_name].train_model()
                        else:
                            self.agents[env_name].train_model()
                else:
                    current_state = next_state[ list(next_state.keys())[0] ]
                    done = next_state[ list(next_state.keys())[0] ].done
                    current_state_name = env.get_state_name()
                    self.set_agent( current_state_name )

            if back_test_env is not None:
                self.back_test_(back_test_env, e, save_best, save_freq, save_dir, save_name)

        pass

    def back_test_(self, back_test_env, i_episode, save_best, save_freq, save_dir = None, save_name = None):

        back_test_score, deals_statistics = self.use_agent(back_test_env, render_deals=True)
        print("Back-test score: {}".format( back_test_score ))

        if back_test_score >= self.best_score:
            print("New best score: {} | Previous best: {}".format( back_test_score, self.best_score ))
            self.best_score = back_test_score
            if save_best:
                self.save_agent( save_dir, save_name )

        if save_freq is not None and save_best is False:
            if save_freq == 0: raise ValueError("Save frequency is 0")
            if i_episode % save_freq == 0:

                #pytorch
                memories = {}
                for agent_name in self.agents.keys():
                    memories[agent_name] = self.agents[agent_name].memory
                    self.agents[agent_name].memory = None

                save( self, os.path.join(save_dir, save_name + "_{}.pkl".format(i_episode)) )

                for agent_name in self.agents.keys():
                    self.agents[agent_name].memory = memories[agent_name]

                #keras
                #self.save_agent(save_dir, save_name + "_{}".format(i_episode))

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
            #self.print_info(env)

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

    def print_info(self, env):
        if env.i_step % 300 == 0:
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
        for agent_name in self.agents.keys():
            memories[agent_name] = self.agents[agent_name].memory
            self.agents[agent_name].memory = None

        for agent_name in self.agents.keys():
            self.agents[agent_name].save_agent( path, name + "_" + agent_name )

        for agent_name in self.agents.keys():
            self.agents[agent_name].memory = memories[agent_name]

        pass

    def load_agent(self, path, name, dropSupportModel = False, compile=True):

        #pytorch
        agent = load( os.path.join( path, name + ".pkl" ) )
        self.agents = agent.agents
        del agent
        gc.collect()

        #keras
        #for agent_name in self.agents.keys():
        #    self.agents[agent_name] = self.agents[agent_name].load_agent( path, name + "_" + agent_name,
        #                                                                dropSupportModel=dropSupportModel,
        #                                                                compile=compile)

        self.n_steps = {"opener": 0, "buy": 0, "sell": 0}
        self.agent = "opener"

        return self