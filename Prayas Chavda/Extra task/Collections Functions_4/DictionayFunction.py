# Dictionary functions
# {} -json format
# it will consider dictionary format
# in key pair value
# general function

dict = {'Name': 'Abc', 'Age': 18}
print(dict)
print(len(dict))

# asit = {'Name': 'Abc', 'Age': 18}
# print(type(asit))
# # # <class 'dict'>
# #
a = str(dict)
print(a[2])
print(type(a))
# # {'Name': 'Abc', 'Age': 18}
# #
# function of dictionary
dict.clear()
print(dict)
# # # clear the dictionary
print(len(dict))
# # 0
# #
dict1 = {'Name': 'Abc', 'Age': 7}
abc = dict1.copy()
print("Copy Data",abc)
# # copy whole dictionary
# # {'Name': 'Abc', 'Age': 7}
# #
tupl = ('name','abc')
the = dict.fromkeys(tupl)
print(the)
# # # # {'name': None, 'age': None}
# # # # in the formkeys it convert data from tupl to dictionary
# # # # in above its only key so it will return none value
# # #
dict = dict.fromkeys(tupl, 10)
print(dict)
# # # {'name': 10, 'age': 10}
# # # if we pass its value it will consider in all keys as same value
# #
dict = {'Name': 'abc', 'Age': 7}
print(dict.get('Age'))
# # # 7
# # # base on key we et the value using finctoin of get
# #
print(dict.get('Education', "Never"))
# # # Never
# # # if we pass the key which are not exist the it return never
# #
dict = {'Name': 'Abc', 'Age': 7}
print(dict.items())
# # ls=dict_items([('Name', 'Abc'), ('Age', 7)])
# # in the items functoin base on key we get the value in tuple in list
# #
print(dict.values())
# dict_values(['Abc', 7])
# # # in this we get the value of keys
# #
print(dict.keys())
# # # dict_keys(['Name', 'Age'])
# # # in this we get the keys
# #
# #
dict1 = {'Name': 'xyz', 'Age': 7}
dict2 = {'Gender': 'male'}
dict1.update(dict2)
print(dict1)
print(dict2)
# # {'Name': 'xyz', 'Age': 7, 'Gender': 'male'}
