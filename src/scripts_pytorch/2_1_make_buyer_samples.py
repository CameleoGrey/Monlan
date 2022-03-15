
################################################################
# Getting best composite agent from previous step and generating
# training samples for future supervised opener by opening
# buy deal in each time step of training data.
################################################################

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


start_time = datetime.now()
print("start time: {}".format(start_time))

symbol = "EURUSD_i"
timeframe = "M10"
feat_list = ["open", "close", "low", "high", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
data_updater = SymbolDataUpdater("../../data/raw/")
data_manager = SymbolDataManager("../../data/raw/")

#data_updater.full_update(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = data_manager.get_data(symbol, timeframe)
df = df.tail(380000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True)
########


################################
# train agent
################################
price_diff_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
opener_price_diff_generator = price_diff_generator
buyer_feat_gen = price_diff_generator
seller_price_diff_generator = price_diff_generator

reward_transformer = load( os.path.join("../../models/reward_transformer.pkl") )

test_df = mod_df.tail(340000).head(300000)
composite_env = CompositeEnv(test_df, opener_price_diff_generator, buyer_feat_gen, seller_price_diff_generator,
                       start_deposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                       stop_type="const", take_type="const", stop_pos=2, take_pos=2, max_loss=20000, max_take=20000,
                       stoploss_puncts=2000, takeprofit_puncts=2000, risk_points=110, risk_levels=5, parallel_opener=False,
                       render_dir="../../data/pictures", render_name="test_plot", env_name="test_env", turn_off_spread=False)
buyer_env = composite_env.state_dict["buyer"]

agent = load( os.path.join("../../models/", "composite_agent_6.pkl"))
buyer_agent = agent.agents["buyer"]
buyer_agent.epsilon = 0.0

history_price_array = composite_env.history_price
price_feature_names_dict = composite_env.price_feature_names_dict

steps_i = []
observations = []
close_rewards = []
start_point = composite_env.get_start_point()
for i in tqdm(range(start_point, len(test_df) - 2), desc="Collecting buyer statistics"):
    open_obs = buyer_feat_gen.get_features(history_price_array, price_feature_names_dict, i)
    action = buyer_agent.get_action( open_obs )
    buyer_env.set_open_point( i )
    state_descriptor = buyer_env.step(history_price_array, price_feature_names_dict, i, action)

    j = i
    while state_descriptor.action != 1:
        j += 1
        if j == len(test_df) - 2:
            break

        next_obs = buyer_feat_gen.get_features(history_price_array, price_feature_names_dict, j)
        action = buyer_agent.get_action( next_obs )
        state_descriptor = buyer_env.step(history_price_array, price_feature_names_dict, j, action)

    if j == len(test_df) - 2:
        break

    reward = state_descriptor.reward_dict[1]
    steps_i.append( i )
    observations.append( open_obs )
    close_rewards.append( reward )

steps_i = np.array( steps_i )
observations = np.array( observations )
close_rewards = np.array( close_rewards )

buyer_samples = { "id": steps_i, "x": observations, "y": close_rewards}
save( buyer_samples, os.path.join("../../data/interim", "buyer_samples.pkl") )
print("done")