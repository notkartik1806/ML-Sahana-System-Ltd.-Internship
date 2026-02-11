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
    
    def add_student(self, roll_no, name, subjects_marks=None):
        student = Student(roll_no, name)
        if subjects_marks:
            for subject, marks in subjects_marks.items():
                student.add_subject_marks(subject, marks)
        self.students.append(student)
    
    def get_data(self, count):
        for i in range(count):
            roll_no = int(input(f"Enter roll number for student {i+1}: "))
            name = input(f"Enter name for student {i+1}: ")
            self.add_student(roll_no, name)
    
    def add_subjects(self, sub_count):
        for student in self.students:
            print(f"\nEnter marks for {student.name}:")
            for j in range(sub_count):
                subject = input(f"Enter subject name {j+1}: ")
                marks = int(input(f"Enter marks for {subject}: "))
                student.add_subject_marks(subject, marks)
    
    def write_data(self, filename="Student_info.txt"):
        with open(filename, "w") as file:
            for i, student in enumerate(self.students, 1):
                file.write(f"Student {i}:\n")
                for key, value in student.get_student_dict().items():
                    file.write(f"{key}: {value}\n")
    
    def display_all_students(self):
        for i, student in enumerate(self.students, 1):
            print(f"\nStudent {i}:")
            for key, value in student.get_student_dict().items():
                print(f"  {key}: {value}")
    
    def save_to_file(self, filename="Student_info.txt"):
        self.write_data(filename)
    
    def display_students(self):
        self.display_all_students()