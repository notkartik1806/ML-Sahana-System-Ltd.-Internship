
def getdata(student_data, count):
    for i in range(count):
        temp_dict = {}
        temp_dict["roll_no"] = int(input("enter roll number: "))
        temp_dict["name"] = input("enter name: ")
        student_data.append(temp_dict)
    print(student_data)

def add_sub(student_data, count, subjects, sub_count):
    for i in range(sub_count):
        subjects.append(input(f"Enter Subject {i} name: "))
    print(subjects)
    for i in student_data:
        print(f"enter {i["name"]} details")
        for sub in subjects:
            i[f"{sub}"] = int(input(f"enter marks for {sub} subject: "))
    print(student_data)

def write_data(student_data, count):
    file = open("Student_info.txt" , "w")
    iteration = 1
    for obj in student_data:
        file.write(f"Student {iteration} data:\n")
        for key, value in obj.items():
                file.write(f"{key} : {value} \n")
        iteration += 1
    file.close()

