# function basic
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'
    
for x in [-1, 0, 1]:
    print(sign(x))

# function with optional keyword arguments
def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('hello, %s' % name.capitalize())

hello('alleysark')
hello('alleysark', True)
hello('alleysark', loud=True)

# class
# Greeter is derived from `object` class
class Greeter(object):
    # constructor
    def __init__(self, name):
        self.name = name # create an instance variable
    
    # instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('hello, %s' % self.name.capitalize())

g = Greeter('alleysark')
g.greet()
g.greet(loud=True)