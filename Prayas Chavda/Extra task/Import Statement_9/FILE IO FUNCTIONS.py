# os module/library

import os

# rename()	=>	It is used to rename a file. It takes two arguments, existing_file_name and new_file_name.

os.rename(existing_file_name, new_file_name)

# remove()	=>	It is used to delete a file. It takes one argument.
#				Pass the name of the file which is to be deleted as the argument of method.

os.remove(filename)

# mkdir()	=>	It is used to create a directory. A directory contains the files.
#				It takes one argument which is the name of the directory.

os.mkdir(path)

# chdir()	=>	It is used to change the current working directory.
#				It takes one argument which is the name of the directory.

os.chdir(path)

# getcwd()	=>	It gives the current working directory.

os.getcwd()

# rmdir()	=>	It is used to delete a directory.
# 				It takes one argument which is the name of the directory.

os.rmdir(path)

# listdir()	=>	Return a list of the entries in the directory given by path.

os.listdir(path)