

import os
import numpy as np
from src.monlan.utils.save_load import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

common_samples = load( os.path.join("../../data/interim", "common_samples.pkl") )
x = common_samples["x"]
y = common_samples["y"]

#clean outliers
buyer_quantile = np.quantile(np.abs(y[:, 0]), q=0.99)
seller_quantile = np.quantile(np.abs(y[:, 1]), q=0.99 )
x = x[ np.abs(y[:, 0]) < buyer_quantile  ]
y = y[ np.abs(y[:, 0]) < buyer_quantile  ]
x = x[ np.abs(y[:, 1]) < seller_quantile  ]
y = y[ np.abs(y[:, 1]) < seller_quantile  ]


scaler = MinMaxScaler(feature_range=(-1, 1))
scale_y = y.reshape( (y.shape[0] * y.shape[1], 1) )
#sns.displot( scale_y )
#plt.show()
scaler.fit( scale_y )
y[:, 0] = scaler.transform( y[:, 0].reshape( (-1, 1) ) ).reshape((-1,))
y[:, 1] = scaler.transform( y[:, 1].reshape( (-1, 1) ) ).reshape((-1,))

# remove near null
"""x = x[ np.abs(y[:, 0]) > 0.1  ]
y = y[ np.abs(y[:, 0]) > 0.1  ]
x = x[ np.abs(y[:, 1]) > 0.1  ]
y = y[ np.abs(y[:, 1]) > 0.1  ]"""

train_x, test_x, train_y, test_y = train_test_split( x, y, shuffle=False, test_size=0.05, random_state=45 )
train_x, val_x, train_y, val_y = train_test_split( train_x, train_y, shuffle=False, test_size=0.05, random_state=46 )

train_dataset = {"x": train_x, "y": train_y}
val_dataset = {"x": val_x, "y": val_y}
test_dataset = {"x": test_x, "y": test_y}

save( train_dataset, os.path.join("../../data/processed", "train_dataset.pkl") )
save( val_dataset, os.path.join("../../data/processed", "val_dataset.pkl") )
save( test_dataset, os.path.join("../../data/processed", "test_dataset.pkl") )

print("done")