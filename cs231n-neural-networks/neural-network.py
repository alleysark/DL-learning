import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

''' generating data '''
Nperc = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
N = Nperc * K
X = np.zeros((N, D)) # data matrix (each row = single example)
y = np.zeros(N, dtype='uint8') # class labels

for j in range(K):
    r = np.linspace(0.0, 1.0, Nperc) # radius
    t = np.linspace(j * 4, (j + 1) * 4, Nperc) + np.random.randn(Nperc)*0.2 # theta
    ix = range(Nperc*j, Nperc*(j + 1))
    X[ix] = np.column_stack([r * np.sin(t), r * np.cos(t)])
    y[ix] = j

#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=matplotlib.cm.get_cmap('Spectral'))
#plt.show()

''' training a softmax linear classifier '''
W = 0.01 * np.random.randn(D, K) # initialize weights randomly
b = np.zeros((1, K)) # biases are safe to be zeroed

step_size = 1
reg_lambda = 1e-3 # regularization strength

for i in range(200):
    ''' compute the class scores '''
    scores = np.dot(X, W) + b # it is [N X K] dim. matrix

    ''' compute the loss '''
    # use cross-entropy loss (i.e. associated with the softmax classifier)
    # p_i = exp(f_yi) / sum(exp(f_j) for all classes j)
    # L_i = -log( p_i )
    # L = sum(L_i for all data i) / N  +  0.5 * lambda * sum(W^2)
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    sum_exp_scores[np.abs(sum_exp_scores) < 1e-5] = 1e-5 #prevent divide-by-zero
    props = np.divide(exp_scores, sum_exp_scores)
    correct_logprobs = np.negative(np.log(props[range(N), y]))
    data_loss = np.sum(correct_logprobs) / N

    regularization_loss = 0.5 * reg_lambda * np.sum(W**2)
    loss = data_loss + regularization_loss

    if i % 10 == 0:
        print('iteration %d: loss %f' % (i, loss))
    
    ''' computing the analytic gradient with backpropagation '''
    dscores = props
    dscores[range(N), y] -= 1
    dscores /= N

    dW = np.dot(X.T, dscores) + reg_lambda*W
    db = np.sum(dscores, axis=0, keepdims=True) # i can't understand this..

    ''' performing a parameter update '''
    W += -step_size * dW
    b += -step_size * db

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
accuracy = np.mean(predicted_class == y)
print('training accuracy: %.2f' % (accuracy))