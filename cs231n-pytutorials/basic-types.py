# number
print("--- number ---")
x = 3
print(type(x), x)
y = 2.5
print(type(y), y)

# boolean
print("--- boolean ---")
t = True
f = False
print(type(t), t, type(f), f)
print(t and f)
print(t or f)
print(not t)
print(t != f) #logical xor

# strings
print("--- strings ---")
hello = 'hello'
world = 'world'
print(hello)
print(len(hello))
print(hello + ' ' + world) # string concatenation
hw12 = '%s %s %d' % (hello, world, 12) # sprintf style string formatting
print(hw12)

s = "hello"
print(s.capitalize())
print(s.upper())
print(s.rjust(7)) # right-justify a string
print(s.center(7)) # center a string
print(s.replace('l', '(ell)'))
print("  world ".strip())