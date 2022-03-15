

from classes.delta_bender.ChangePredictor import ChangePredictor
from classes.delta_bender.FeatGen_CDV import FeatGen_CDV
from classes.delta_bender.PlotRender import PlotRender
from classes.delta_bender.SymbolDataManager import SymbolDataManager
from classes.multipurpose.utils import *
from classes.multipurpose.DenseAutoEncoder import DenseAutoEncoder
from tqdm import tqdm


symbol = "EURUSD_i"
timeframe = "M5"
target_columns = ["open", "high", "low", "close", "tick_volume"]
dataManager = SymbolDataManager("../data/raw/")
df = dataManager.getData(symbol, timeframe)
df = df[ target_columns ]
df = df.tail(600)
df_vals = df.values
print(df.shape)

window_size = 200
ema_period = 64
n_price_feats = 10
m_cdv_feats = 44
use_heiken_ashi = True
batch_size = 128

model = load( "../models/ex_stack.pkl" )
#encoder = DenseAutoEncoder().loadEncoder( "../models/", "encoder" )
change_predictor = ChangePredictor( use_heiken_ashi, ema_period, n_price_feats, m_cdv_feats,
                                    model, None, batch_size)

pred_probas = []
for i in tqdm( range( len(df_vals) - window_size), desc="generating dataset", colour="green" ):

    x_i = df_vals[i: i + window_size]
    x_i = pd.DataFrame( x_i, columns=target_columns )
    y_pred = change_predictor.predict_proba( x_i )
    y_pred = list( y_pred )
    pred_probas.append( y_pred )
stub_vals = [[0.0, 0.0] for i in range(window_size)]
pred_probas = stub_vals + pred_probas
pred_probas = np.array( pred_probas )
df["assurance_bull"] = pred_probas[:, 1]
df["assurance_bear"] = pred_probas[:, 0]

df = FeatGen_CDV().transform(df, period=ema_period)

PlotRender().plot_assurance(df)

print("done")


