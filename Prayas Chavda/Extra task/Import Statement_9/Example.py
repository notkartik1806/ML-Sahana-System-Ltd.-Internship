# The import Statement

# You can use any Core_Python source file as a module by executing an import statement in some other Core_Python source file.

# import module1[, module2[,... moduleN]

# 1
import Module1 as M1

M1.fn1("Softvan")

# The from...import Statement

# Core_Python's from statement lets you import specific attributes from a module into the current namespace.

# from modname import name1[, name2[, ... nameN]]
# 2
from Module2 import fn1

fn1(10, 15)

# The from...import * Statement

# It is also possible to import all names from a module into the current namespace

# from modname import *
# 3

from Module3 import *

fn1(10, 5)

fn2("Hello Core_Python")
