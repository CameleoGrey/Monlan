from trafficante.Environment import Environment
import numpy as np

env = Environment(stSize=10)
env.setBarData("EURUSD", "H1")
scaler = env.getScaler()

basePrice = env.getBasePrice()
doneFlag = False
while doneFlag == False:
    st, yt, doneFlag = env.step()
    print( str(st) + " " + str( (yt - basePrice) / basePrice) + " " + str(doneFlag) )