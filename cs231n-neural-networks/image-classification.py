import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from cs231n import data_utils
import numpy as np

class NearestNeighbor(object):
    def __init__(self, dist = 'L1'):
        if dist == 'L1':
            self.dist = self.L1_distance
        elif dist == 'L2':
            self.dist = self.L2_distance
        else:
            raise Warning("invalid distance function name!")
    
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
                distances.append( self.dist(Xtr_row, X[i, :]) )
            min_idx = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_idx]
        
        return Ypred
    
    def L1_distance(self, Xtr, Xte):
        return np.sum(np.abs(Xtr - Xte))

    def L2_distance(self, Xtr, Xte):
        return np.sqrt(np.sum(np.square(Xtr - Xte)))


def classify_images_simple(Xtr, Ytr, Xte, Yte):
    nn = NearestNeighbor()
    nn.train(Xtr, Ytr)
    Yte_predict = nn.predict(Xte)

    print('accuracy: %f' % (np.mean(Yte_predict == Yte)))

def classify_images_with_validation_set(Xtr, Ytr, val_len):
    Xval = Xtr[:val_len, :] # take first `val_len` for validation
    Yval = Ytr[:val_len]
    Xtr = Xtr[val_len:, :] # keep others for training
    Ytr = Ytr[val_len:]

    validation_accuracies = []
    for dist_func in ['L1', 'L2']:
        nn = NearestNeighbor(dist=dist_func)
        nn.train(Xtr, Ytr)
        Yval_predict = nn.predict(Xval)
        acc = np.mean(Yval_predict == Yval)
        print('accuracy: %f' % (acc))

        validation_accuracies.append((dist_func, acc))


Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10('../cs231n/datasets/cifar-10-batches-py/')
Xtr = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

classify_images_simple(Xtr, Ytr, Xte, Yte)

classify_images_with_validation_set(Xtr, Ytr, val_len=1000)