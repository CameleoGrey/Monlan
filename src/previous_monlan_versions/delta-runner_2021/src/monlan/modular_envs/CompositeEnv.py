

from src.monlan.modular_envs.OpenerEnv import OpenerEnv
from src.monlan.modular_envs.BuyerEnv import BuyerEnv
from src.monlan.modular_envs.SellerEnv import SellerEnv
from src.monlan.modular_envs.StateDescriptor import StateDescriptor

import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class CompositeEnv():

    def __init__(self, historyPrice, openerFeatureGen, buyerFeatureGen, sellerFeatureGen, startDeposit = 300,
                 lot_size=0.1, lot_coef=100000, spread = 18, spread_coef = 0.00001, stopType = "const", takeType = "const",
                 stoplossPuncts=100, takeprofitPuncts=500, stopPos = 0, takePos = 0, maxLoss = 40000, maxTake = 40000,
                 riskPoints=110, riskLevels=5, parallelOpener = False, renderFlag=True, renderDir=None, renderName=None, verbose=True,
                 env_name=None, turn_off_spread=False ):

        self.env_name = env_name
        self.price_feature_names_dict = {}
        price_feature_names = list(historyPrice.columns)
        for i in range(len(price_feature_names)):
            self.price_feature_names_dict[ price_feature_names[i]] = i
        self.historyPrice = historyPrice.copy().values
        self.iStep = 0
        self.turn_off_spread = turn_off_spread

        #########
        # only FeatGen_ColumnsCompressor
        #openerFeatureGen.cache_vectors(self.historyPrice, self.price_feature_names_dict, self.env_name, batch_size=32768)
        #########

        self.state_dict = {}
        self.state_dict["opener"] = OpenerEnv(openerFeatureGen, spread_coef, lot_coef, lot_size, self.turn_off_spread)
        self.state_dict["buyer"] = BuyerEnv(buyerFeatureGen, spread_coef, lot_coef, lot_size, self.turn_off_spread)
        self.state_dict["seller"] = SellerEnv(sellerFeatureGen, spread_coef, lot_coef, lot_size, self.turn_off_spread)

        self.state_dict["buyer"].set_limit_orders( takeprofit_puncts=takeprofitPuncts, stoploss_puncts=stoplossPuncts )
        self.state_dict["seller"].set_limit_orders( takeprofit_puncts=takeprofitPuncts, stoploss_puncts=stoplossPuncts )

        self.action_space = {}
        self.observation_space = {}
        for state_name in self.state_dict.keys():
            self.action_space[ state_name ] = self.state_dict[state_name].get_action_space()
            self.observation_space[ state_name ] = self.state_dict[state_name].get_feature_shape()

        self.state = self.state_dict["opener"]

        self.done = False
        self.startDeposit = startDeposit
        self.deposit = self.startDeposit
        self.lot_size = lot_size
        self.lot_coef = lot_coef
        self.spread = spread
        self.spread_coef = spread_coef

        self.sumLoss = 0
        self.open_point_id = None
        self.close_point_id = None

        self.verbose = verbose

        # render data
        self.render_dir = renderDir
        self.base_name = renderName
        self.xData = []
        self.openData = []
        self.highData = []
        self.lowData = []
        self.closeData = []
        self.took_actions = []

    def set_state_(self, env_name):

        self.state = self.state_dict[env_name]

        pass

    def get_start_point(self):
        feature_shape = self.state_dict["opener"].feature_generator.get_feature_shape()
        window_width = np.max(list(feature_shape))
        start_point = window_width - 1
        return start_point

    def get_name(self):
        return self.env_name

    def get_state_name(self):
        return self.state.get_name()

    def reset(self):
        self.openPoint = None
        self.sumLoss = 0
        self.done = False
        self.deposit = self.startDeposit
        self.open_point_id = None
        self.close_point_id = None

        #########
        # only FeatGen_ColumnsCompressor
        #for state_name in self.state_dict.keys():
        #    self.state_dict[state_name].feature_generator.set_active_env( self.env_name )
        #########

        self.xData = []
        self.openData = []
        self.highData = []
        self.lowData = []
        self.closeData = []
        self.took_actions = []

        self.iStep = self.get_start_point()

        self.xData.append(self.iStep + 1)
        self.openData.append(self.historyPrice[self.iStep + 1][self.price_feature_names_dict["open"]])
        self.highData.append(self.historyPrice[self.iStep + 1][self.price_feature_names_dict["high"]])
        self.lowData.append(self.historyPrice[self.iStep + 1][self.price_feature_names_dict["low"]])
        self.closeData.append(self.historyPrice[self.iStep + 1][self.price_feature_names_dict["close"]])
        self.took_actions.append("opener_hold")

        self.state = self.state_dict["opener"]
        opener_state_descriptor = self.state.step( self.historyPrice,
                                  self.price_feature_names_dict,
                                  self.iStep,
                                  1 )
        state_descriptors = {}
        state_descriptors["opener"] = opener_state_descriptor

        return state_descriptors

    def step(self, action):

        self.iStep += 1
        state_descriptors = {}
        action_name = self.state.action_space.actionsDict[action]

        if self.iStep + 2 >= len(self.historyPrice):
            state_descriptors["opener"] = StateDescriptor(None, None, None, None, True, None)
            state_descriptors["buyer"] = StateDescriptor(None, None, None, None, True, None)
            state_descriptors["seller"] = StateDescriptor(None, None, None, None, True, None)
            return state_descriptors

        self.xData.append(self.iStep + 1)
        self.openData.append(self.historyPrice[self.iStep + 1][self.price_feature_names_dict["open"]])
        self.highData.append(self.historyPrice[self.iStep + 1][self.price_feature_names_dict["high"]])
        self.lowData.append(self.historyPrice[self.iStep + 1][self.price_feature_names_dict["low"]])
        self.closeData.append(self.historyPrice[self.iStep + 1][self.price_feature_names_dict["close"]])

        if self.state.get_name() == "opener":
            opener_state_descriptor = self.state.step( self.historyPrice, self.price_feature_names_dict, self.iStep, action )
            state_descriptors["opener"] = opener_state_descriptor
            self.state.set_open_point( None )
            if action_name == "buy":
                self.state.set_open_point( self.iStep + 1 )
                self.set_state_( "buyer" )
                self.state.set_open_point( self.iStep + 1 )
                self.took_actions.append("opener_buy")
            elif action_name == "sell":
                self.state.set_open_point( self.iStep + 1 )
                self.set_state_( "seller" )
                self.state.set_open_point( self.iStep + 1 )
                self.took_actions.append("opener_sell")
            else:
                self.took_actions.append("opener_hold")

        elif self.state.get_name() == "buyer":
            buyer_state_descriptor = self.state.step( self.historyPrice, self.price_feature_names_dict, self.iStep, action )
            state_descriptors["buyer"] = buyer_state_descriptor

            if state_descriptors["buyer"].info["limit_triggered"]:
                action_name = "buy"

            if action_name == "buy":
                self.state.set_open_point( None )
                self.set_state_( "opener" )
                self.state.set_close_point( self.iStep + 1 )
                opener_state_descriptor = self.state.get_opener_close_deal_state(self.historyPrice, self.price_feature_names_dict, deal_type="buy")
                state_descriptors["opener"] = opener_state_descriptor

                if state_descriptors["buyer"].info["limit_triggered"]:
                    state_descriptors["opener"].info["deposit_delta"] = state_descriptors["buyer"].reward_dict[1]

                self.state.set_open_point( None )
                self.state.set_close_point( None )
                self.deposit += state_descriptors["opener"].info["deposit_delta"]
                self.took_actions.append("buyer_close")
            else:
                self.took_actions.append("buyer_hold")

        elif self.state.get_name() == "seller":
            seller_state_descriptor = self.state.step( self.historyPrice, self.price_feature_names_dict, self.iStep, action )
            state_descriptors["seller"] = seller_state_descriptor

            if state_descriptors["seller"].info["limit_triggered"]:
                action_name = "sell"

            if action_name == "sell":
                self.state.set_open_point( None )
                self.set_state_( "opener" )
                self.state.set_close_point( self.iStep + 1 )
                opener_state_descriptor = self.state.get_opener_close_deal_state(self.historyPrice, self.price_feature_names_dict, deal_type="sell")
                state_descriptors["opener"] = opener_state_descriptor

                if state_descriptors["seller"].info["limit_triggered"]:
                    state_descriptors["opener"].info["deposit_delta"] = state_descriptors["seller"].reward_dict[1]

                self.state.set_open_point( None )
                self.state.set_close_point( None )
                self.deposit += state_descriptors["opener"].info["deposit_delta"]
                self.took_actions.append("seller_close")
            else:
                self.took_actions.append("seller_hold")
        else:
            raise ValueError("Unknown env")

        if self.verbose and \
            action_name in ["buy", "sell"] and \
            self.state.get_name() == "opener":
            print("Step: {} | Action: {} | Deposit: {}".format( self.iStep, action_name, self.deposit ))


        return state_descriptors

    def render_episode(self):

        for i in tqdm(range(len(self.xData)), desc="Rendering episode"):
            color = None
            linewidth = 0.1
            tail_width = linewidth / 4
            action = self.took_actions[i]
            if action == "opener_hold":
                color = "black"
            elif action == "opener_buy":
                color = "green"
            elif action == "opener_sell":
                color = "red"
            elif action == "buyer_hold":
                color = "cyan"
            elif action == "buyer_close":
                color = "green"
            elif action == "seller_hold":
                color = "yellow"
            elif action == "seller_close":
                color = "red"
            else:
                raise ValueError("Unrecognized action {}".format(action))

            if self.openData[i] < self.closeData[i]:
                plt.plot([self.xData[i], self.xData[i]], [self.lowData[i], self.openData[i]], c="black",linewidth=tail_width)
                plt.plot([self.xData[i], self.xData[i]], [self.closeData[i], self.highData[i]], c="black", linewidth=tail_width)
            else:
                plt.plot([self.xData[i], self.xData[i]], [self.lowData[i], self.closeData[i]], c="black", linewidth=tail_width)
                plt.plot([self.xData[i], self.xData[i]], [self.openData[i], self.highData[i]], c="black", linewidth=tail_width)

            plt.plot([self.xData[i], self.xData[i]], [self.openData[i], self.closeData[i]], c=color, linewidth=linewidth)


        save_path = os.path.join( self.render_dir, self.base_name + "_{}.png".format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) )
        print("Saving image to {}".format(save_path))
        plt.savefig(save_path, dpi=3000)
        plt.cla()
        plt.clf()
        pass