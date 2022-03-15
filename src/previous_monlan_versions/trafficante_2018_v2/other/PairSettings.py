


class PairSettings:

    def __init__(self):

        self.availablePairs = ["EURUSD", "EURJPY", "USDJPY"]
        self.availablePeriods = ["M5", "M10", "M15", "M30", "H1"]

        self.pairName = "EURUSD"
        self.pairPeriod = "H1"

        self.domainColumns = ["<CLOSE>", "<OPEN>", "<LOW>", "<HIGH>"]
        self.targetColumns = ["<CLOSE>"]
        self.plotReal      = [True]
        self.plotPredict   = [True]
        self.L = 100
        self.P = 1
        self.epochs = 30
        self.batch_size = 50
        self.split_coef = 0.9

        self.N = len( self.domainColumns )
        self.targetColumnsCount = len( self.targetColumns )

        self.workDataParams = {}
        self.workDataParams["inputSeqCount"] = self.N
        self.workDataParams["inputSeqLength"] = self.L
        self.workDataParams["predictSeqLength"] = self.P

        pass

    def setPairName(self, name, period):

        self.pairName = name
        self.pairPeriod = period

        pass

    def setDomainColumns(self, domainColumns=["<OPEN>", "<CLOSE>", "<LOW>", "<HIGH>"]):

        self.domainColumns = ["<OPEN>", "<CLOSE>", "<LOW>", "<HIGH>"]
        self.N = len( domainColumns )

        pass

    def setTargetColumns(self, targetColumns=["<OPEN>", "<CLOSE>", "<LOW>", "<HIGH>"]):

        self.targetColumns = ["<OPEN>", "<CLOSE>", "<LOW>", "<HIGH>"]
        self.targetColumnsCount = len( targetColumns )

        pass

    def setTrainParams(self, L = 200, P = 12, epochs = 10, batch_size = 50, split_coef = 0.97):

        self.L = L
        self.P = P
        self.epochs = epochs
        self.batch_size = batch_size
        self.split_coef = split_coef

        self.workDataParams["inputSeqLength"] = L
        self.workDataParams["predictSeqLength"] = P

        pass

