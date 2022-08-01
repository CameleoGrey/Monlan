
#################################################################
# Script for training buyer and seller agents with random opener.
# Training opener by Reinforcement Learning leads to overfitting:
# opener will always try to open buy or sell deals because one of
# them was leading to more profit at the WHOLE history. Opener
# must be myopic (good at short term) and buyer/seller must be
# visionary ("do I need hold the deal to get more profit or close
# to avoid losses?").
#################################################################


from src.monlan.modular_agents.DQNAgent_pytorch import DQNAgent
from src.monlan.modular_agents.CompositeAgent import CompositeAgent
from src.monlan.modular_agents.RewardTransformer import RewardTransformer
from src.monlan.modular_agents.EmaRewardTransformer import EmaRewardTransformer
from src.monlan.modular_envs.CompositeEnv import CompositeEnv
from src.monlan.feature_generators.FeatGen_CDV import FeatGen_CDV
from src.monlan.feature_generators.FeatGen_ScaledWindow import FeatGen_ScaledWindow
from src.monlan.datamanagement.SymbolDataManager import SymbolDataManager
from src.monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from src.monlan.terminal.MT5Terminal import MT5Terminal
from src.monlan.utils.save_load import *
from datetime import datetime
import os


start_time = datetime.now()
print("start time: {}".format(start_time))

symbol = "EURUSDrfd"
timeframe = "M10"
feat_list = ["open", "close", "low", "high", "tick_volume", "delta", "cdv"]

#terminal = MT5Terminal(login=123456, server="broker-server", password="password")
data_updater = SymbolDataUpdater("../../../../../data/raw/")
data_manager = SymbolDataManager("../../../../../data/raw/")

#data_updater.full_update(terminal, symbol, timeframe, start_date="2010-01-01 00:00:00")
df = data_manager.get_data(symbol, timeframe, normalize_names = False, normalize_date_time = False)
df = df.tail(380000)

########
mod_df = FeatGen_CDV().transform(df, period=14, verbose=True, add_raw_delta=True)
########


################################
# train agent
################################
opener_price_diff_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
buyer_price_diff_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)
seller_price_diff_generator = FeatGen_ScaledWindow(feat_list, n_points=256, flat_stack=False)

#reward_transformer = RewardTransformer()
#reward_transformer.fit(df, spread_coef=0.00001, lot_coef=100000, lot_size=0.1, window_width=40)
reward_transformer = EmaRewardTransformer()
reward_transformer.fit(df, spread_coef=0.00001, lot_coef=100000, lot_size=0.1, deal_length=60, history_step_size=10, averaging_n=40)
save(reward_transformer, path=os.path.join("../../../../../models/reward_transformer.pkl"))
reward_transformer = load(os.path.join("../../../../../models/reward_transformer.pkl"))

train_df = mod_df.tail(340000).head(300000)
train_env = CompositeEnv(train_df, opener_price_diff_generator, buyer_price_diff_generator, seller_price_diff_generator,
                         start_deposit=1000.0, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                         stop_type="const", take_type="const", stop_pos=2, take_pos=1, max_loss=20000, max_take=20000,
                         stoploss_puncts=2000, takeprofit_puncts=2000, risk_points=110, risk_levels=5, parallel_opener=False,
                         render_flag=True, render_dir="../../../../../data/pictures", render_name="train_plot", env_name="train_env", turn_off_spread=True)

back_test_df = mod_df.tail(380000).head(40000)
back_test_env = CompositeEnv(back_test_df, opener_price_diff_generator, buyer_price_diff_generator, seller_price_diff_generator,
                             start_deposit=1000, lot_size=0.1, lot_coef=100000, spread=18, spread_coef=0.00001,
                             stop_type="const", take_type="const", stop_pos=2, take_pos=2, max_loss=20000, max_take=20000,
                             stoploss_puncts=2000, takeprofit_puncts=2000, risk_points=110, risk_levels=5, parallel_opener=False,
                             render_dir="../../../../../data/pictures", render_name="back_plot", env_name="back_env", turn_off_spread=False)

# get size of state and action from environment
opener_agent = DQNAgent( "opener", train_env.observation_space["opener"], train_env.action_space["opener"].n,
                           memory_size=20, batch_size=16, train_start=50, epsilon_min=1.0, epsilon=1, discount_factor=0.0,
                           epsilon_decay=0.9999, learning_rate=0.0001, reward_comparison_steps=40, reward_scaler=reward_transformer)
buyer_agent = DQNAgent( "buyer", train_env.observation_space["buyer"], train_env.action_space["buyer"].n,
                          memory_size=300000, batch_size=16, train_start=270000, epsilon_min=0.05, epsilon=1, discount_factor=0.95,
                          epsilon_decay=0.99999, learning_rate=0.0001, reward_comparison_steps=40, reward_scaler=reward_transformer)
seller_agent = DQNAgent( "seller", train_env.observation_space["seller"], train_env.action_space["seller"].n,
                           memory_size=300000, batch_size=16, train_start=270000, epsilon_min=0.05, epsilon=1, discount_factor=0.95,
                           epsilon_decay=0.99999, learning_rate=0.0001, reward_comparison_steps=40, reward_scaler=reward_transformer)

agent = CompositeAgent(opener_agent, buyer_agent, seller_agent, start_deposit=1000.0)
agent.fit_agent(env=train_env, back_test_env=back_test_env, n_episodes=200, n_warm_up=0,
                uniform_eps=False, syn_eps=False, plot_scores=False,
                save_best=False, save_freq=1, save_dir="../../../../../models/", save_name="kaiming_composite_agent",
                excluded_from_training_agents=["opener"])

end_time = datetime.now()
print("Training finished. Total time: {}".format(end_time - start_time))