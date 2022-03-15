
import numpy as np
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from classes.delta_bender.FeatGen_PriceEMA import FeatGen_PriceEMA
from classes.delta_bender.FeatGen_HeikenAshi import FeatGen_HeikenAshi
from sklearn.preprocessing import MinMaxScaler

class DBAgentOpener_HACDV:
    def __init__(self, agentName, model):

        self.name = agentName
        self.batch_size = 200
        self.epsilon = 0
        self.memory = []
        self.nPoints = 200
        self.ema_period_volume = 14
        self.window_size = 64

        self.fg_sw = FeatGen_ScaledWindow(["open", "high", "low", "close", "cdv"], nPoints=self.window_size, nDiffs=0,
                                          flatStack=False)
        self.fg_cdv = FeatGen_CDV()
        self.fg_pe = FeatGen_PriceEMA()

        self.model = model

    def get_heiken_cdv_obs_(self, df):
        df = df.tail(self.nPoints).copy()
        df = FeatGen_HeikenAshi().transform(df, verbose=False)

        df = self.fg_cdv.transform(df, self.ema_period_volume)
        df = df.iloc[self.ema_period_volume - 1:]
        df = df.tail(self.window_size)
        del df["tick_volume"]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        for feat in ["open", "high", "low", "close"]:
            tmp = df[feat].values.reshape((-1, 1))
            scaler.partial_fit(tmp)
        for feat in ["open", "high", "low", "close"]:
            tmp = df[feat].values.reshape((-1, 1))
            tmp = scaler.transform(tmp)
            tmp = tmp.reshape((-1,))
            df[feat] = tmp

        for feat in ["cdv"]:
            tmp = df[feat].values.reshape((-1, 1))
            tmp = MinMaxScaler(feature_range=(-1, 1)).fit_transform(tmp)
            tmp = tmp.reshape((-1,))
            df[feat] = tmp

        # PlotRender().plot_price_cdv(df)

        # obs = obsList.values
        obs = np.vstack([df["open"].values, df["cdv"].values,
                        df["high"].values, df["cdv"].values,
                        df["low"].values, df["cdv"].values,
                        df["close"].values, df["cdv"].values, ])
        obs = obs.reshape((1,) + obs.shape + (1,))

        return obs

        # get action from model using epsilon-greedy policy

    def get_action(self, df):

        df = df.copy()
        df = df[["open", "high", "low", "close", "tick_volume"]]

        x_i = self.get_heiken_cdv_obs_(df)
        y_pred = self.model.predict(x_i)[0]

        bear_proba = y_pred[0]
        bull_proba = y_pred[1]

        proba_treshold = 0.65
        #if bull_proba > bear_proba and bull_proba > proba_treshold:
        #    action = 0
        if bull_proba < bear_proba and bear_proba > proba_treshold:
            action = 2
        else:
            action = 1


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