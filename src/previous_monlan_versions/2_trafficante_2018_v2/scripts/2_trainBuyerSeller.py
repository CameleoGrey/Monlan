

from trafficante.CloserWaiter import CloserWaiter

pair = "EURUSD"
period = "H1"

closerWaiter = CloserWaiter(dealTypes="buy")
#closerWaiter.loadScaler(pair, period)
closerWaiter.train(pair, period, epochs = 100, maxFrames = 6, stSize = 100)
closerWaiter.saveModel()
closerWaiter.saveScaler( pair, period )

closerWaiter = CloserWaiter(dealTypes="sell")
#closerWaiter.loadScaler(pair, period)
closerWaiter.train(pair, period, epochs = 100, maxFrames = 6, stSize = 100)
closerWaiter.saveModel()
closerWaiter.saveScaler( pair, period )
