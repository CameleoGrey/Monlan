import os
import numpy as np
from src.monlan.utils.save_load import *
import torch
import torchvision.models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Hinge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

train_dataset = load( os.path.join("../../data/processed", "train_vector_dataset.pkl") )
test_dataset = load( os.path.join("../../data/processed", "test_vector_dataset.pkl") )
val_dataset = load( os.path.join("../../data/processed", "val_vector_dataset.pkl") )

train_x, train_y = train_dataset["x"], train_dataset["y"]
val_x, val_y = val_dataset["x"], val_dataset["y"]

"""model = CatBoostRegressor(n_estimators=10000,
                          thread_count=16,
                          max_depth=4,
                          #task_type="GPU",
                          early_stopping_rounds=2000,
                          loss_function="MultiRMSE"
                          )
model.fit( train_x, train_y[:, 0], eval_set=(val_x, val_y[:, 0]) )"""

#model = ExtraTreesRegressor(n_estimators=100, max_depth=30, n_jobs=6)
model = LinearRegression(n_jobs=10)
model.fit( train_x, train_y[:, 0] )
pred_y = model.predict( val_x )
mae = mean_absolute_error(val_y[:, 0], pred_y)
mse = mean_squared_error(val_y[:, 0], pred_y)
print( mae )
print( mse )


"""train_y_clf = train_y[:, 0] > train_y[:, 1]
train_y_clf = train_y_clf.astype(int)


val_y_clf = val_y[:, 0] > val_y[:, 1]
val_y_clf = val_y_clf.astype(int)

model = CatBoostClassifier(n_estimators=10000,
                          thread_count=16,
                          max_depth=8,
                          task_type="GPU",
                          early_stopping_rounds=5000,
                          )
model.fit( train_x, train_y_clf, eval_set=(val_x, val_y_clf) )
"""