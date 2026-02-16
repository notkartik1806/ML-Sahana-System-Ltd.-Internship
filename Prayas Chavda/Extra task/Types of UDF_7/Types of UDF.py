#   Function Arguments:

#   1)  Required arguments
#   2)  Keyword arguments
#   3)  Default arguments
#   4)  Variable-length arguments
#	5)	Dictionary arguments
#   6)  Lamda Function


#   1) Required arguments:

# Required arguments are the arguments passed to a function in correct positional order.
# Here, the number of arguments in the function call should match exactly with the function definition.
# we have to pass the argument base on requirement in parameter.
# Positional Argument
def fn1(c):
    # c=a+b
    print(c)


fn1("Hello World")
# fn1(10,20)

# Output:Hello World


#   2) Keyword arguments:

# Keyword arguments are related to the function calls.
# When you use keyword arguments in a function call,
# the caller identifies the arguments by the parameter name.


# Keyword arguments (or named arguments) are values that, when passed into a function, are identifiable by specific parameter names.
#  A keyword argument is preceded by a parameter and the assignment operator, = .

def fn2(a,b):
    print(a,b)


fn2(b="Good Evening",a="hi")


# Output:Good Evening

def team(name, project):
    print(name, "is working on an", project)

team(project = "Edpresso", name = 'FemCode')


#   3)  Default arguments:

#       A default argument is an argument that assumes a default value
#       if a value is not provided in the function call for that argument,it prints default value if it is not passed

#default argument means it takes the value of parameter as default argument
# e.g 
def my_function(country = "Norway"):
  print("I am from " + country)

my_function("Sweden")
my_function("India")
my_function()
my_function("Brazil")

#output

# I am from Sweden
# I am from India
# I am from Norway --default argument
# I am from Brazil


def fn3(name, marks=35):
    print("Name=", name)

    print("Marks=", marks)


fn3(marks=50, name="XYZ")

# Output:

#	Name=XYZ
#	Marks=50

fn3(name="ABC")


# Output:

#	Name=ABC
#	Marks=35


#   4)  Variable-length arguments

# You may need to process a function for more arguments than you specified while defining the function.
# These arguments are called variable-length arguments and are not given in the function definition,

# An asterisk (*) is placed before the variable name that holds the values of all nonkeyword variable arguments.
#This tuple remains empty if no additional arguments are specified during the function call.
#it makes the function flexible.

def fn4(*a):
    print(type(a))
    print(a)

fn4(50)
# Output:50
fn4("string", 70, "Hello", "disha", "rushika", "smit")


# Output:
#	60
#	70
#	Hello


#	5)	Dictionary arguments

# #A keyword argument is where you provide a name to the variable as you pass it into the function.
# #One can think of the kwargs as being a dictionary that maps each keyword to the value that we pass alongside it.
# #That is why when we iterate over the kwargs there doesn’t seem to be any order in which they were printed out.
#in this argument create dictionary pass number of methods to allowingperform number of dictinary operations.
def fn5(**std):
    print(std)
    print(type(std))
    if std is not None:
        for key, value in std.items():
            # print("{} = {}".format(key, value)) or
            print("%s = %s" % (key, value))

fn5(fn='Abc', ln='Def', name="Yash", demo="Hemil")

# Output:
#	fn=Abc
#	ln=Def

# lambda function=>anonymous function means without name (as we use to define function with keyword def bt lamda is itself keyword use to define function)\

# This function can have any number of arguments but only one expression, which is evaluated and returned.
# One is free to use lambda functions wherever function objects are required.
# You need to keep in your knowledge that lambda functions are syntactically restricted to a single expression.
# It has various uses in particular fields of programming, besides other types of expressions in functions.

#A lambda function can take any number of arguments, but can only have one expression.
"""
to execute your single line business logic
"""
square = lambda x: x * x
print(square(15))

addition = lambda x, y: x + y
print(addition(15, 25))

old_list = [1, 2, 3, 4, 5]
# ls = []
# for i in old_list:
#     ls.append(i * 2)

new_list = list(map(lambda x: x * 2, old_list))
print(new_list)

old_list = [1, 2, 3, 4, 5]
new_list = list(filter(lambda x: (x % 2 == 0), old_list))
print(new_list)




#for map

# Parameter	Description
# function -	Required. The function to execute for each item
# iterable -	Required. A sequence, collection or an iterator object. You can send as many iterables as you like, just make sure the function has one parameter for each iterable.



# for fiter

# Filter() is a built-in function in Python. The filter function can be applied to an iterable such as a list or tuple and create a new iterator.
#  This new iterator can filter out certain specific elements based on the condition that you provide very efficiently.