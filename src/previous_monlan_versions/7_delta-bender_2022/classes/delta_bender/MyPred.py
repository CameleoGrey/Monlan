
import numpy as np

class MyPred:
    def __init__(self):
        pass

    def predict(self, x):

        y_pred = []
        for i in range( len(x) ):
            y_i = x[i][-1]
            y_i = 1 if y_i >= 0 else -1
            y_pred.append( y_i )
        y_pred = np.array( y_pred )

        return y_pred