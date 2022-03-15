
################################################################
# Merging partial datasets from previous step into one to get
# estimations (targets) of buyer and seller performance in each
# point of training data.
################################################################


import os
import numpy as np
from tqdm import tqdm
from src.monlan.utils.save_load import *

buyer_samples = load( os.path.join("../../data/interim", "buyer_samples.pkl") )
seller_samples = load( os.path.join("../../data/interim", "seller_samples.pkl") )

buyer_steps = buyer_samples["id"]
buyer_x = buyer_samples["x"]
buyer_y = buyer_samples["y"]

seller_steps = seller_samples["id"]
seller_x = seller_samples["x"]
seller_y = seller_samples["y"]

common_x = []
common_y = []
for i in tqdm(range(len(buyer_steps)), desc="Building common dataset"):
    if buyer_steps[i] != seller_steps[i]:
        raise ValueError("Different step_i")
    if buyer_x[i][0][0][0][0] != seller_x[i][0][0][0][0]:
        raise ValueError("Different obs")

    common_x.append( buyer_x[i] )
    target = [ buyer_y[i], seller_y[i] ]
    common_y.append( target )

common_x = np.array( common_x )
common_y = np.array( common_y )

common_x = common_x.reshape((common_x.shape[0], common_x.shape[1], common_x.shape[2], common_x.shape[3]))

common_samples = { "x": common_x, "y": common_y }

save( common_samples, os.path.join( "../../data/interim", "common_samples.pkl" ) )
print( "done" )


