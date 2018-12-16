# list
print("===== list =====")
xs = [3, 1, 2]
print(type(xs), xs, xs[2]) # python uses 0-based indexing
print(xs[-1]) # negative indices count from the end of the list
xs[2] = 'foo' # list can contain elements of different types
print(xs)

xs.append('bar') # add (append) new element(s)
x = xs.pop()
print(x, xs)

## slicing
print("--- slicing ---")
nums = list(range(5)) # make list from 'range'
print(nums)
print(nums[2:4]) # get a slice from index 2 to 4 (exclusive)
print(nums[2:]) # get a slice from index 2 to the end
print(nums[:2]) # get a slice from the start to index 2
print(nums[:]) # get a slice of the whole list. it means that it makes a copy of list
print(nums[:-1]) # slice indices can be negative
nums[2:4] = [8, 9] # assign a new sublist to a slice

## loops
print("--- loops ---")
animals = ['cat', 'doc', 'monkey']
for animal in animals:
    print(animal)

for idx, animal in enumerate(animals):
    print('%d: %s' % (idx, animal))

## list comprehensions
print("--- list comprehensions ---")
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)

squares = [x ** 2 for x in nums]
print(squares)

even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)


# dictionaries
print("===== dictionaries =====")
d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat']) # get entry from a dictionary
print('cat' in d) # check if a dictionary has a given key
d['fish'] = 'wet'
print(d['fish'])
#print(d['monkey'])  KeyError
print(d.get('monkey', 'N/A'), ' ', d.get('fish', 'N/A'))
del d['fish'] #remove an element from a dictionary
print(d)

## loops
print("--- loops ---")
d = {'person': 2, 'cat': 4, 'dog': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('%s has %d legs' % (animal, legs))

for animal, legs in d.items():
    print('%s has %d legs' % (animal, legs))

## dictionary comprehensions
print("--- dictionary comprehensions ---")
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)

four_leg_animals = {k: v for k, v in d.items() if v == 4}
print(four_leg_animals)

# sets - unordered collection of distinct elements
print("===== sets =====")
animals = {'cat', 'dog'}
print('cat' in animals, ', ', 'fish' in animals) # check if an element is in a set
animals.add('fish')
print(len(animals), animals)
animals.add('cat') # adding an element that is already in the set does nothing
print(len(animals), animals)
animals.remove('cat') # remove an element from a set
print(len(animals), animals)

## loops
print("--- loops ---")
for idx, animal in enumerate(animals):
    print('%d: %s' % (idx, animal))

## set comprehensinos
print("--- set comprehinsions ---")
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)

# tuples - (immutable) ordered list of values
print("===== tuples =====")
d = {(x, x + 1) : x for x in range(5)} # create a dictionary with tuple keys
print(d)
t = (4, 5)
print(type(t))
print(d[t])
print(d[(1, 2)])

nth_tuple = (1, 2.0, 'three') # tuple with different types, more that two
print(nth_tuple)