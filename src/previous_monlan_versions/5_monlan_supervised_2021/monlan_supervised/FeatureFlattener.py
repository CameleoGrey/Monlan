
from monlan_supervised.DenseAutoEncoder import DenseAutoEncoder
from tqdm import tqdm
import numpy as np

class FeatureFlattener():
    def __init__(self):
        self.colEncoder = None
        self.vecEncoder = None
        pass

    def makeColumnsVecSet(self, x):

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))
        colVecs = np.zeros((x.shape[0] * x.shape[2], x.shape[1], 1), dtype=np.float32)
        for i in tqdm(range(len(x)), desc="Making columns set"):
            vecBatch = x[i].T
            vecBatch = vecBatch.reshape( vecBatch.shape + (1,) )
            for j in range(len(vecBatch)):
                colVecs[i * len(vecBatch) + j] = vecBatch[j]
        return colVecs

    def buildColEncoder(self, colDim, lr):
        self.colEncoder = DenseAutoEncoder().buildModel(colDim, latentDim=1, lr=lr)
        return self

    def fitColEncoder(self, x, batchSize, epochs):
        self.colEncoder.fit(x, batchSize, epochs)
        return self

    """def flattenizeFeats(self, x):
        flattenedFeats = []
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))
        for i in tqdm(range(len(x)), desc="Flattenising feats"):
            vecBatch = x[i].T
            vecBatch = vecBatch.reshape(vecBatch.shape + (1,))
            featVec = self.colEncoder.encode(vecBatch)
            flattenedFeats.append(featVec)
        flattenedFeats = np.array(flattenedFeats)
        return flattenedFeats"""

    def flattenizeFeats(self, x, verbose=1):
        flattenedFeats = []
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))
        if verbose >= 1:
            for i in tqdm(range(len(x)), desc="Flattenising feats"):
                vecBatch = x[i].T
                vecBatch = vecBatch.reshape(vecBatch.shape + (1,))
                flattenedFeats.append( vecBatch )
        else:
            for i in range(len(x)):
                vecBatch = x[i].T
                vecBatch = vecBatch.reshape(vecBatch.shape + (1,))
                flattenedFeats.append( vecBatch )
        flattenedFeats = np.vstack(flattenedFeats)
        flattenedFeats = self.colEncoder.predict(flattenedFeats, batch_size=4096, verbose=verbose)
        flattenedFeats = np.array_split(flattenedFeats, len(x))
        flattenedFeats = np.array(flattenedFeats)
        return flattenedFeats

    def buildVecEncoder(self, inputVecDim, latentDim, lr):
        self.vecEncoder = DenseAutoEncoder().buildModel(inputVecDim, latentDim=latentDim, lr=lr)
        return self

    def fitVecEncoder(self, x, batchSize, epochs):
        self.vecEncoder.fit(x, batchSize=batchSize, epochs=epochs)
        return self

    def compressVecs(self, x, batchSize, verbose=1):
        x = self.vecEncoder.predict(x, batch_size=batchSize, verbose=verbose)
        return x

    def saveFlattener(self, dir, name):
        self.colEncoder.saveEncoder(dir, name + "columns")
        self.vecEncoder.saveEncoder(dir, name + "flatvecs")
        return self

    def loadFlattener(self, dir, name):
        self.loadColEncoder(dir, name)
        self.loadVecEncoder(dir, name)
        return self

    def saveColEncoder(self, dir, name):
        self.colEncoder.saveEncoder(dir, name + "columns")
        return self

    def saveVecEncoder(self, dir, name):
        self.vecEncoder.saveEncoder(dir, name + "flatvecs")
        return self

    def loadColEncoder(self, dir, name):
        self.colEncoder = DenseAutoEncoder().loadEncoder(dir, name + "columns")
        return self

    def loadVecEncoder(self, dir, name):
        self.vecEncoder = DenseAutoEncoder().loadEncoder(dir, name + "flatvecs")
        return self
