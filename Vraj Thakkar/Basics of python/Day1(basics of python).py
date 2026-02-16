info_file = open("studentinfo.txt", "w")
marks_file = open("studentmarks.txt", "w")
subject_file = open("subject.txt", "w")

# Number of students
n = int(input("Enter the number of students: "))

students = []

# Student details
for i in range(n):
    rollno = int(input("Enter the roll number: "))
    name = input("Enter the name: ")

    stu = {"rollno": rollno, "name": name}
    students.append(stu)

info_file.write(str(students))
print(students)

# Number of subjects
m = int(input("Enter the number of subjects: "))

subjects = []

for j in range(m):
    sub = input("Enter the subject name: ")
    subjects.append(sub)

subject_file.write(str(subjects))

# Marks dictionary (student-wise)
marks = {}

# Initialize roll numbers
for stu in students:
    marks[stu["rollno"]] = {}

# Enter marks
for stu in students:
    for sub in subjects:
        rollno = stu["rollno"]
        score = int(input(f"Enter the marks of {stu['name']} in {sub}: "))
        marks[rollno][sub] = score

marks_file.write(str(marks))

print(marks)

