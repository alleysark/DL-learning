import numpy as np

# arrays
print("===== arrays =====")
a = np.array([1, 2, 3])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
a[0] = 5
print(a)

## array creation functions
print("--- array creation functions ---")
print(np.zeros((2, 2)))
print(np.ones((2, 2)))
print(np.full((2, 2), 7))
print(np.eye(2)) # 2x2 identity matrix
print(np.random.random((2, 2)))

# array indexing
print("===== array indexing =====")
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = a[:2, 1:3] # first two rows and columns 1 and 2
print(b)

b = a[::-1, ::2]
print(b)

# a slice of an array ia a view into the same data
# so modifying it will modify the original array
b[0, 0] = 99
print(a)

print("--- integer array indexing ---")
# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the original array
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)

# integer array indexing allows you to construct arbitrary arrays 
# using the data from another array
a = np.array([[1, 2], [3, 4], [5, 6]])

b = a[[0, 1, 2], [1, 1, 0]] # data of a where (row 0, col 1), (row 1, col 1), (row 2, col 0)
print(b)

b = a[[0, 0, 1, 1, 2, 2], 1] # it seems like swizzling
print(b)

## useful trick
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
b = np.array([0, 2, 0, 1]) # indices array
print(a[np.arange(4), b])
a[np.arange(4), b] += 10
print(a)

## boolean array indexing
print("--- boolean array indexing ---")
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
bool_idx = (a % 2 == 0) # it returns numpy.ndarray of booleans of the same shape as `a`
print(bool_idx, type(bool_idx))
print(a[bool_idx])
print(a[a % 2 == 0]) # it is same with above
a[a % 2 == 0] = 0
print(a)

# data types
print("===== data types =====")
x = np.array([1, 2])
print(x.dtype)

x = np.array([1.0, 2.0])
print(x.dtype)

x = np.array([1, 2], dtype=np.int64) # force a particular datatype
print(x.dtype)

# array math
print("===== array math =====")
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y) # operator '*' is not matrix multiplication
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))

print(x ** 0.5)
print(np.sqrt(x))

## vector / matrix product
v = np.array([1, 2])
w = np.array([3, 4])
u = np.array([[3], [4]])
print(v.dot(w), np.dot(v, w)) # inner product of vectors
print(v.dot(u)) # rank of the result is different with above

print(x.dot(v), np.dot(x, v)) # matrix - vector product. it produces the rank 1 array

print(x.dot(y)) # matrix - matrix proudct. it produces the rank 2 array
print(np.dot(x, y))

## array sums
print(np.sum(x)) # compute sum of all elements
print(np.sum(x, axis=0)) # compute sum of each column
print(np.sum(x, axis=1)) # compute sum of each row

## array manipulations. 
# for more informations: https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html
print(x)
print(x.T) # transpose

v = np.array([1,2,3])
print(v.T) # note that taking the transpose of a rank 1 does nothing

z = np.stack([x, y]) # make stacked array into one np.ndarray data
print(z)
print([x, y]) # it is different with above

## broadcasting
# [this post](http://sacko.tistory.com/16) is more helpful to understand.
print("===== broadcasting =====")
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x) # create an empty matrix with the same shape as x

for i in range(4):
    y[i, :] = x[i, :] + v # add v to each row of the matrix x. it could be slow for large matrix
print(y)

vv = np.tile(v, (4, 1)) # stack 4 copies of v on top of each other
y = x + vv # add x and vv elementwise
print(y)

y = x + v # it actually work same as above. thanks to the numpy broadcasting.
print(y) # it does not make copies of v and it does not iterate over elements!

v = np.array([1, 2, 3])
w = np.array([4, 5])
print(np.reshape(v, (3, 1)) * w) # make v to 3x1 column matrix, and broadcast toward axis-1
                                 # it is same with np.array([[1, 1], [2, 2], [3, 3]]) * w