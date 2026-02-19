int_list=[]
str_list=[]


#Taking input from user for variable
a1=int(input("Enter the number of elements you want to add in variable: "))
a=[i for i in input("Enter the elements(seperated bhy comma): ").split(",") ]

for i in range(a1):    
    if a[i].isnumeric():
        int_list.append(int(a[i]))
    else:
        str_list.append(a[i])

#Printing List 1 and List 2
print("List 1 (int_list) is: ", int_list)
print("List 2 (str_list) is: ", str_list)

#Finding the maximum and minimum value in List 1 (int_list)
print("Max of List 1 is: ", max(int_list))
print("Min of List 1 is: ", min(int_list))

print("Reversed List 2 is: ", list(reversed(str_list)))
