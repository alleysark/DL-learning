import numpy as np
import math

# f(x, y) = {x + sigmoid(y)} / {sigmoid(x) + (x + y)**2}
x = 3
y = 4

def sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))

# forward pass
a = sigmoid(y)
b = x + a
c = sigmoid(x)
d = x + y
e = d**2
f = c + e
g = 1.0 / f
h = b * g

# backward pass (reverse mode differentiation)
dhdb = g
dhdg = b

dgdf = -1.0 / (f**2)

dfdc = 1
dfde = 1

dedd = 2.0 * d

dddx = 1
dddy = 1

dcdx = (1 - c) * c

dbdx = 1
dbda = 1

dady = (1 - a) * a

# chaining all partial differentiations
dhdx = dhdg * dgdf * (dfdc * dcdx + dfde * dedd * dddx) + dhdb * dbdx
dhdy = dhdg * dgdf * (dfde * dedd * dddy) + dhdb * dbda * dady

print (x, y, dhdx, dhdy)