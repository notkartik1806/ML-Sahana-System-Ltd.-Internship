from functions import *
student_data = []

count = int(input("enter count: "))

getdata(student_data, count)

subjects = []

sub_count = int(input("enter subject count: "))

add_sub(student_data, count, subjects, sub_count)

write_data(student_data, count)

