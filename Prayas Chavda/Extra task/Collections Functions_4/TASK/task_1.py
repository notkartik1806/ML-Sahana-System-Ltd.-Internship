var = input("Enter input values: ")

str_list = []
int_list = []

splittedvar = var.split(',')

for item in splittedvar:
    if item.isdigit():
        int_list.append(int(item))
    else:
        str_list.append(item)
    
if int_list:
    print(f"Int list: {int_list}")
    print(f"Min: {min(int_list)}")
    print(f"Max: {max(int_list)}")

if str_list:
    print(f"Str list: {str_list}")
    print(f"Reversed: {str_list[::-1]}")