
from src.monlan.modular_agents.DQNAgent_keras import DQNAgent
from src.monlan.modular_agents.CompositeAgent import CompositeAgent
from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from datetime import datetime
import os
import numpy as np
from src.monlan.utils.save_load import *
from tqdm import tqdm

import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


startTime = datetime.now()
print("start time: {}".format(startTime))

symbol = "EURUSD_i"
timeframe = "M10"
hkFeatList = ["open", "close", "low", "high", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
dataUpdater = SymbolDataUpdater("../../data/raw/")
dataManager = SymbolDataManager("../../data/raw/")

#dataUpdater.fullUpdate(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = dataManager.getData(symbol, timeframe)
df = df.tail(400000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True)
########


################################
# train agent
################################
priceDiffGenerator = FeatGen_ScaledWindow(hkFeatList, nPoints=256, flatStack=False)
opener_feat_gen = priceDiffGenerator
buyer_feat_gen = priceDiffGenerator
seller_feat_gen = priceDiffGenerator

reward_transformer = load( os.path.join("../../models/reward_transformer.pkl") )

testDf = mod_df.tail(140000).tail(1000)
composite_env = CompositeEnv(testDf, opener_feat_gen, buyer_feat_gen, seller_feat_gen,
                             startDeposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                             stopType="const", takeType="const", stopPos=2, takePos=2, maxLoss=20000, maxTake=20000,
                             stoplossPuncts=200, takeprofitPuncts=100, riskPoints=110, riskLevels=5, parallelOpener=False,
                             renderDir="../../data/pictures", renderName="test_plot", env_name="test_env", turn_off_spread=False)
seller_env = composite_env.state_dict["seller"]
openerAgent = DQNAgent("opener", composite_env.observation_space["opener"], composite_env.action_space["opener"].n)
buyerAgent = DQNAgent("buyer", composite_env.observation_space["buyer"], composite_env.action_space["buyer"].n)
sellerAgent = DQNAgent("seller", composite_env.observation_space["seller"], composite_env.action_space["seller"].n)
agent = CompositeAgent(openerAgent, buyerAgent, sellerAgent, reward_transformer)
agent  = agent.load_agent("../../models/", "composite_agent_0")
#agent  = agent.load_agent("../../models/", "checkpoint_composite_0")
seller_agent = agent.agents["seller"]
seller_agent.epsilon = 0.0

history_price_array = composite_env.historyPrice
price_feature_names_dict = composite_env.price_feature_names_dict

steps_i = []
observations = []
close_rewards = []
start_point = composite_env.get_start_point()
for i in tqdm(range(start_point, len(testDf) - 2, 10), desc="Collecting seller statistics"):
    open_obs = seller_feat_gen.get_features(history_price_array, price_feature_names_dict, i)
    action = seller_agent.get_action( open_obs )
    seller_env.set_open_point( i )
    state_descriptor = seller_env.step(history_price_array, price_feature_names_dict, i, action)

    j = i
    while state_descriptor.action != 1:
        j += 1
        next_obs = seller_feat_gen.get_features(history_price_array, price_feature_names_dict, j)
        action = seller_agent.get_action( next_obs )
        state_descriptor = seller_env.step(history_price_array, price_feature_names_dict, j, action)

    reward = state_descriptor.reward_dict[1]
    steps_i.append( i )
    observations.append( open_obs )
    close_rewards.append( reward )

steps_i = np.array( steps_i )
observations = np.array( observations )
close_rewards = np.array( close_rewards )

seller_samples = { "id": steps_i, "x": observations, "y": close_rewards}
save( seller_samples, os.path.join("../../data/interim", "seller_samples.pkl") )
seller_samples = load( os.path.join("../../data/interim", "seller_samples.pkl") )
print(seller_samples)
print("done")