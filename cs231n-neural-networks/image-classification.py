import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from cs231n import data_utils
import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        # X is N x D where each row is an example. Y is 1d lables of size N
        # We just keep this without any complicated training
        self.Xtr = X
        self.ytr = y
    
    def predict(self, X):
        # X is N x D where each row is an example we wish to predict label for
        num_test = X.shape[0]

        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance
            print("now predict... (%.3f)" % (float(i) / num_test))
            distances = []
            for Xtr_row in self.Xtr:
                distances.append(np.sum(np.abs(Xtr_row - X[i, :])))
            min_idx = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_idx]
        
        return Ypred


Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10_noreshape('../cs231n/datasets/cifar-10-batches-py/') # a magic function we provide

nn = NearestNeighbor()
nn.train(Xtr, Ytr)
Yte_predict = nn.predict(Xte)

print('accuracy: %f' % (np.mean(Yte_predict == Yte)))