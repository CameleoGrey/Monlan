
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
from mlxtend.classifier import StackingCVClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

np.random.seed(45)

x = load( "../data/x_for_model.pkl" )
y = load( "../data/y_for_model.pkl" )

#########
encoder = DenseAutoEncoder().loadEncoder("../models/", "encoder")
x = x.reshape(x.shape + (1,))
x = encoder.predict(x, batch_size=128)
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

print("fit catboost stack")
clfList = [
    KNeighborsClassifier(n_neighbors=2, leaf_size=30, n_jobs=8),
    CatBoostClassifier( n_estimators=1000,
                           max_depth=4,
                           thread_count=8,
                           task_type="CPU",
                        logging_level="Silent",
                           auto_class_weights="SqrtBalanced",
                           random_state=45),
    XGBClassifier(n_jobs=8, verbosity=1, random_state=12),
    LGBMClassifier(n_jobs=8, random_state=13),
    ExtraTreesClassifier( n_estimators=1000, max_depth=2, n_jobs=8 ),
    ExtraTreesClassifier( n_estimators=1000, max_depth=4, n_jobs=8 ),
    ExtraTreesClassifier( n_estimators=1000, max_depth=10, n_jobs=8 ),
    ExtraTreesClassifier( n_estimators=1000, max_depth=30, n_jobs=8 ),
]
lr = ExtraTreesClassifier( n_estimators=1000, max_depth=30, n_jobs=8 )

model = StackingCVClassifier(classifiers=clfList,
                            use_probas=True,
                            use_features_in_secondary=True,
                            meta_classifier=lr,
                            cv=20,
                            random_state=15, verbose=1)

model.fit(x_train, y_train)
#y_pred = sclf.predict(x_test)
#score(y_pred, y_test)


#model = load("../models/catboost_model.pkl")
y_pred = model.predict( x_val )
scores = get_all_scores(y_pred, y_val)
print( scores )
y_pred = model.predict( x_test )
scores = get_all_scores(y_pred, y_test)
print( scores )
probas_test = model.predict_proba(x_test)
save( model, "../models/ex_stack.pkl" )


print("done")

