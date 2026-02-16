class StudentManager:
    def __init__(self):
        self.student_data = []
        self.subjects = []

    def getdata(self, count):
        for i in range(count):
            temp_dict = {}
            temp_dict["roll_no"] = int(input("Enter roll number: "))
            temp_dict["name"] = input("Enter name: ")
            self.student_data.append(temp_dict)

        print(self.student_data)

    def add_sub(self, sub_count):
        for i in range(sub_count):
            self.subjects.append(input(f"Enter Subject {i+1} name: "))

        print(self.subjects)

        for student in self.student_data:
            print(f'Enter {student["name"]} details')
            for sub in self.subjects:
                student[sub] = int(input(f"Enter marks for {sub}: "))

        print(self.student_data)

    def write_data(self):
        file = open("Student_info.txt", "w")

        i = 1
        for obj in self.student_data:
            file.write(f"Student {i} data:\n")
            for key, value in obj.items():
                file.write(f"{key} : {value}\n")
            i += 1

        file.close()