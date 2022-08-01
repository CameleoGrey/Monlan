
from classes.delta_bender.FeatGenMeta import FeatGenMeta
from classes.multipurpose.utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from classes.multipurpose.DenseAutoEncoder import DenseAutoEncoder
from classes.delta_bender.MyPred import MyPred
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

np.random.seed(45)

x = load( "../data/x_for_model.pkl" )
y = load( "../data/y_for_model.pkl" )

#########
#encoder = DenseAutoEncoder().loadEncoder("../models/", "encoder")
#x = x.reshape(x.shape + (1,))
#x = encoder.predict(x, batch_size=128)
#########

#x_train, x_test, y_train, y_test = train_test_split( x, y, stratify=y, test_size=0.2, shuffle=True, random_state=45 )

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, shuffle=False, random_state=45 )
x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, stratify=y_train, test_size=0.2, shuffle=True, random_state=45 )

"""tsne = TSNE( n_jobs=8, verbose=1 )
x_tsne = tsne.fit_transform( x_train[:2000] )
y_uniq = np.unique( y_train )
for yu in y_uniq:
    plt.scatter( x_tsne[ y_train[:2000] == yu, 0],  x_tsne[ y_train[:2000] == yu, 1] )
plt.show()"""

model = MyPred()

#model = load("../models/catboost_model.pkl")
y_pred = model.predict( x_val )
scores = get_all_scores(y_pred, y_val)

print( scores )
y_pred = model.predict( x_test )
scores = get_all_scores(y_pred, y_test)
print( scores )



print("done")

