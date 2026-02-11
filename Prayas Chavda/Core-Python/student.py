class Student:
    def __init__(self, roll_no, name):
        self.roll_no = roll_no
        self.name = name
        self.subjects = {}
    
    def add_subject_marks(self, subject, marks):
        self.subjects[subject] = marks
    
    def get_student_dict(self):
        student_dict = {
            "roll_no": self.roll_no,
            "name": self.name
        }
        student_dict.update(self.subjects)
        return student_dict


class StudentManager:
    def __init__(self):
        self.students = []
        self.subjects = []
    
    def get_data(self, count):
        for i in range(count):
            roll_no = int(input("Enter roll number: "))
            name = input("Enter name: ")
            student = Student(roll_no, name)
            self.students.append(student)
        print("\nStudents added successfully!")
    
    def add_subjects(self, sub_count):
        for i in range(sub_count):
            subject = input(f"Enter Subject {i+1} name: ")
            self.subjects.append(subject)
        print(f"\nSubjects: {', '.join(self.subjects)}")
        
        for student in self.students:
            print(f"\nEnter {student.name}'s details:")
            for subject in self.subjects:
                marks = int(input(f"Enter marks for {subject}: "))
                student.add_subject_marks(subject, marks)
        print("\nAll data collected!")
    
    def write_data(self, filename="Student_info.txt"):
        file = open(filename, "w")
        iteration = 1
        for obj in self.students:
            file.write(f"Student {iteration} data:\n")
            student_dict = obj.get_student_dict()
            for key, value in student_dict.items():
                file.write(f"{key} : {value}\n")
            iteration += 1
        file.close()
        print(f"\nData written to {filename}!")
    
    def display_all_students(self):
        print("\n" + "="*50)
        print("ALL STUDENT DATA")
        print("="*50)
        for idx, student in enumerate(self.students, 1):
            print(f"\nStudent {idx}:")
            student_dict = student.get_student_dict()
            for key, value in student_dict.items():
                print(f"  {key}: {value}")