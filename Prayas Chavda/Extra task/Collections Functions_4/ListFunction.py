# function of list

# general function of list
# length function

# list1, list2 = [123, 'xyz', 'Nimesh'], [456, 'abc']
# print("First list length : ", len(list1))
# print("Second list length : ", len(list2))

# Output
# First list length :  3
# Second list length :  2

# convert tuple value in list
aTuple = (123, 'xyz', 'Nimesh', 'abc')
aList = list(aTuple)
# print(aTuple)
# print("List elements : ", aList)

# Output
# List elements :  [123, 'xyz', 'Nimesh', 'abc']

# min,max functoin
list1 = [123, 45, 3, 2]
print(max(list1))
print(min(list1))

# Output 123
# 2

# Function of list
# 1.append
# append is use to update the data

# aList.append(2009)
print("Updated List : ", aList)

# Output
# Updated List :  [123, 'xyz', 'Nimesh', 'abc', 2009]

# 2. count
# to count how many types the number repeat

aList = [123, 'xyz', 'Nimesh', 'abc', 123]

print("Count for 123 : ", aList.count(123))

# Output
# Count for 123 :  2

# 3.extend
# if we want to update the list we use this function

aList = [123, 'xyz', 'Harshal', 'abc', 123]
bList = [2009, 'Nimesh']
aList.extend(bList)
print("Extended List : ", aList)


# Output
# Extended List :  [123, 'xyz', 'Harshal', 'abc', 123, 2009, 'Nimesh']

# 4. index
# it wll find the position

aList = [123, 'xyz', 'Nimesh', 'abc']
print("Index for Nimesh : ", aList.index('Nimesh'))

# Output
# 2

# 5.insert
# this function help to insert data at which position that will given

aList = [123, 'xyz', 'Nimesh', 'abc']
aList.insert(3, 2009)
# position index and data that we have to insert at the particular index
print("Final List : ", aList)

# Output
# Final List : [123, 'xyz', 'Nimesh', 2009, 'abc']

### Pop's out using index(int argument)
# 6.pop
# it will remove fom last

aList = [123, 'xyz', 'Nimesh', 'abc']
print("A List : ", aList.pop())
print(aList)
# # A List :  abc
# # [123, 'xyz', 'Nimesh']
#
print("B List : ", aList.pop(2))
# if we want to remove from index we have to pass the argument like this
print(aList)
# B List :  Nimesh
# [123, 'xyz']

# 7.remove

aList = [123, 'xyz', 'Nimesh', 'abc', 'xyz']
aList.remove('xyz')
# using this function remove from the list
print("List : ", aList)
# Output
# List :  [123, 'Nimesh', 'abc', 'xyz']

# 8. reverse
# using this function it reverse the string

aList = [123, 'xyz', 'Nimesh', 'abc', 'xyz']
aList.reverse()
print("List : ", aList)

# Output
# List :  ['xyz', 'abc', 'Nimesh', 'xyz', 123]

# 9. sort
# it will sort list in assending order
aList = [47.5, 123, 123456789.66666666]
aList.sort()
print("List : ", aList)

# Output
#  [47.5, 123, 123456789.66666666]
