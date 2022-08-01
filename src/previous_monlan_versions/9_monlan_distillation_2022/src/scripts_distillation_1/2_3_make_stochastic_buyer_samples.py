
################################################################
# Getting best composite agent from previous step and generating
# training samples for future supervised opener by opening
# buy deal in each time step of training data.
################################################################

from src.monlan.modular_agents.StochasticCloser import StochasticCloser
from src.monlan.feature_generators.NoneGenerator import NoneGenerator
from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.feature_generators.FeatGen_HistoryEmbedder import FeatGen_HistoryEmbedder
from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from datetime import datetime
import os
import numpy as np
from src.monlan.utils.save_load import *
from tqdm import tqdm


start_time = datetime.now()
print("start time: {}".format(start_time))

symbol = "EURUSDrfd"
timeframe = "M10"
feat_list = ["open", "close", "low", "high", "tick_volume", "delta", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
data_updater = SymbolDataUpdater("../../../../../data/raw/")
data_manager = SymbolDataManager("../../../../../data/raw/")

#data_updater.full_update(terminal, symbol, timeframe, startDate="2008-01-01 00:00:00")
df = data_manager.get_data(symbol, timeframe)
df = df.tail(380000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True, add_raw_delta=True)
########


################################
# train agent
################################
buyer_agent = StochasticCloser()

base_feat_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
stub_generator = NoneGenerator(feat_list, n_points=256, flat_stack=False)

opener_base_feat_generator = stub_generator
buyer_feat_generator = stub_generator
seller_base_feat_generator = stub_generator

history_df = mod_df.tail(340000).head(300000)
composite_env = CompositeEnv(history_df, opener_base_feat_generator, buyer_feat_generator, seller_base_feat_generator,
                             start_deposit=300, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                             stop_type="const", take_type="const", stop_pos=2, take_pos=2, max_loss=20000, max_take=20000,
                             stoploss_puncts=2000, takeprofit_puncts=2000, risk_points=110, risk_levels=5, parallel_opener=False,
                             render_dir="../../../../../data/pictures", render_name="test_plot", env_name="test_env", turn_off_spread=True)
buyer_env = composite_env.state_dict["buyer"]

history_price_array = composite_env.history_price
price_feature_names_dict = composite_env.price_feature_names_dict

steps_i = []
raw_open_features = []
close_rewards = []
open_price = []
high_price = []
low_price = []
close_price = []
start_point = composite_env.get_start_point()

start_hold_proba = 0.2
start_close_proba = 1.0 - start_hold_proba
end_hold_proba = 0.95
end_close_proba = 1.0 - end_hold_proba
n_estimates = 100
proba_increment = (end_hold_proba - start_hold_proba) / n_estimates

for i in tqdm(range(start_point, len(history_df) - 2), desc="Collecting buyer statistics"):
    open_point_raw_features = base_feat_generator.get_features(history_price_array, price_feature_names_dict, i)
    raw_open_features.append(open_point_raw_features)

    steps_i.append(i)
    buyer_env.set_open_point( i )
    state_descriptor = buyer_env.step(history_price_array, price_feature_names_dict, i, 0)

    open_price.append( history_price_array[i][price_feature_names_dict["open"]] )
    high_price.append( history_price_array[i][price_feature_names_dict["high"]] )
    low_price.append( history_price_array[i][price_feature_names_dict["low"]] )
    close_price.append( history_price_array[i][price_feature_names_dict["close"]] )


    state_value_scores = []
    current_hold_proba = start_hold_proba
    current_close_proba = start_close_proba
    for estimate_i in range( n_estimates ):
        j = i
        while state_descriptor.action != 1:
            j += 1
            if j == len(history_df) - 2:
                break

            action = buyer_agent.get_action_head_only( current_hold_proba, current_close_proba )
            state_descriptor = buyer_env.step(history_price_array, price_feature_names_dict, j, action)

        if j == len(history_df) - 2:
            break

        reward = state_descriptor.reward_dict[1]
        state_descriptor = buyer_env.step(history_price_array, price_feature_names_dict, i, 0)
        state_value_scores.append( reward )
        current_hold_proba += proba_increment
        current_close_proba -= proba_increment

    state_value_score = np.mean(state_value_scores)
    close_rewards.append( state_value_score )

steps_i = np.array( steps_i )
raw_open_features = np.array( raw_open_features )
close_rewards = np.array( close_rewards )

open_price = np.array( open_price )
high_price = np.array( high_price )
low_price = np.array( low_price )
close_price = np.array( close_price )

buyer_samples = { "id": steps_i,
                  "x_raw": raw_open_features,
                  "y": close_rewards,
                  "open": open_price,
                  "high": high_price,
                  "low": low_price,
                  "close": close_price }
save(buyer_samples, os.path.join("../../../../../data/interim", "buyer_samples.pkl"))
print("done")