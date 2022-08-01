
import numpy as np

from keras.models import Model, Input
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

n = 100
m = 10

inputLayer = Input(shape=(n, 1))
conv_1 = Conv1D(100, 3, activation="tanh")(inputLayer)
pool_1 = MaxPool1D()(conv_1)
conv_2 = Conv1D(100, 3, activation="tanh")(pool_1)
pool_2 = MaxPool1D()(conv_2)
conv_3 = Conv1D(100, 3, activation="tanh")(pool_2)
dense_1 = Dense(100, activation="tanh")(Flatten()(conv_3))
outputLayer = Dense(1)(dense_1)
model = Model(inputs=inputLayer, outputs=outputLayer)
model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["mean_absolute_percentage_error"])
model.summary()

data = np.ones( (n, n) )
data = np.expand_dims( data, axis=2 )
y = np.ones( (n, 1) )

model.fit( data, y )

print( model.predict( data ) )