# file io

# file_object = open(file_name,[access_mode])

file = open("demo.txt", "r")

# access_mode = read,write,append

# Accessing Mode

# r=r,r+,rb,rb+
# w=w,w+,wb,wb+
# a=a,a+ab,ab+


file = open("1.txt", "r+")

"""
file must be exist
file pointer begining

r  	==>	reading only.
    (memory location)
The file pointer is placed at the beginning of the file.This is the default mode.

r+ 	==> both reading and writing.
The file pointer placed at the beginning of the file.

rb 	==> reading only in binary format.
The file pointer is placed at the beginning of the file.(image read)

rb+ ==> both reading and writing in binary format.
The file pointer placed at the beginning of the file.

"""

file = open("1.txt", "w+")

'''

w   ==>	writing only.Overwrites the file if the file exists.
		If the file does not exist, creates a new file for writing.

w+  ==>	both writing and reading.Overwrites the existing file if the file exists.
		If the file does not exist, creates a new file for reading and writing.

wb  ==> writing only in binary format.Overwrites the file if the file exists.
		If the file does not exist, creates a new file for writing.

wb+ ==> both writing and reading in binary format.Overwrites the existing file if the file exists.
		If the file does not exist, creates a new file for reading and writing.

'''

file = open("1.txt", "a+")

'''

a	==>	appending. The file pointer is at the end of the file, if the file exists.
		If the file does not exist, it creates a new file for writing.

a+	==>	both appending and reading.The file pointer is at the end of the file, if the file exists.
		If the file does not exist, it creates a new file for reading and writing.

ab	==>	appending in binary format.The file pointer is at the end of the file if the file exists.
		If the file does not exist, it creates a new file for writing.

ab+	==>	both appending and reading in binary format.The file pointer is at the end of the file, if the file exists.
		If the file does not exist, it creates a new file for reading and writing.

'''

#File Functions
_________________
#1. open() # openfile
#2. write() # write file
#3. read() # read file
#4. close() #close the object file. It save the file and free the memory location.
#5. tell() The tell() method tells you the current position of index.  within the file
#6. seek()


# file object

# read a file
fo = open("foo.txt", "r+")
str = fo.read()
position = fo.tell()
# 

# reposition pointer
fo.seek(0, 0)
print("Read String is : ", str)
fo.close()

---------------------------------------------------------
# seek(offset,whence)

# offset means character index
#--start with 1 index.

# whence means position

# 0= begining position
# 1= current position
# 2= ending position

#In this when offset work whnece can not work both can not work at same time.

# write in a file

fo = open("foo.txt", "w")
fo.write("Core_Python is a great language.\nYeah its great!!\n")

fo1 = open("word.txt", "w")
fo1.write("Core_Python is a great language.\nYeah its great!!\n")

fo.close()

