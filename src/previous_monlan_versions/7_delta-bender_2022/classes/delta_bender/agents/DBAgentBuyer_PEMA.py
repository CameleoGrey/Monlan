
import numpy as np
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from classes.delta_bender.FeatGen_PriceEMA import FeatGen_PriceEMA

class DBAgentBuyer_PEMA:
    def __init__(self, agentName, model):

        self.name = agentName
        self.batch_size = 200
        self.epsilon = 0
        self.memory = []

        self.model = model

    # get action from model using epsilon-greedy policy
    def get_action(self, df):

        df = df.copy()
        df = df[["open", "close", "low", "high", "tick_volume"]]

        window_size = 64
        ema_period_volume = 14
        ema_period_price = 3

        df = FeatGen_CDV().transform(df, ema_period_volume)
        df = df.iloc[ema_period_volume - 1:]

        df = FeatGen_PriceEMA().transform(df, ema_period_price, verbose=False)
        df = df.iloc[ema_period_price - 1:]

        del df["tick_volume"]

        feat_gen = FeatGen_ScaledWindow(["open", "high", "low", "close", "cdv"], nPoints=window_size, nDiffs=0, flatStack=False)
        ind = df.index.values[-1]
        x_i = feat_gen.get_window(ind, df)
        y_pred = self.model.predict(x_i)[0]

        bear_proba = y_pred[0]
        bull_proba = y_pred[1]

        proba_treshold = 0.8
        """if bull_proba > bear_proba and bull_proba > proba_treshold:
            action = 0
        elif bear_proba > bear_proba and bear_proba > proba_treshold:
            action = 1
        else:
            action = 0"""

        #if bull_proba > bear_proba and bull_proba > proba_treshold:
        #    action = 0
        #else:
        #    action = 1

        action = 0

        #if bear_proba > proba_treshold:
        #    action = 0
        #else:
        #    action = 1

        return action
    def build_model(self):

        pass

    def fit_agent(self, env, nEpisodes, plotScores, saveFreq=5):

        pass

    def use_agent(self, env):
        pass

    def save_agent(self, path, name):

        pass

    def load_agent(self, path, name, dropSupportModel = False):

        return self

    def update_target_model(self):

        pass

    def loadPretrainedWeights(self, dir, baseName, agentType):

        return self

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        pass

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        pass