student = {
    's1': {
        'rollno': int(input("Enter roll number for s1: ")),
        'name': input("Enter name for s1: "),
        'marks': int(input("Enter marks for s1: ")),
        'grade': None,
    },
    's2': {
        'rollno': int(input("Enter roll number for s2: ")),
        'name': input("Enter name for s2: "),
        'marks': int(input("Enter marks for s2: ")),
        'grade': None,
    },
    's3': {
        'rollno': int(input("Enter roll number for s3: ")),
        'name': input("Enter name for s3: "),
        'marks': int(input("Enter marks for s3: ")),
        'grade': None,
    },
    's4': {
        'rollno': int(input("Enter roll number for s4: ")),
        'name': input("Enter name for s4: "),
        'marks': int(input("Enter marks for s4: ")),
        'grade': None,
    },
    's5': {
        'rollno': int(input("Enter roll number for s5: ")),
        'name': input("Enter name for s5: "),
        'marks': int(input("Enter marks for s5: ")),
        'grade': None,
    }
}

for student_key in student:
    marks = student[student_key]['marks']
    if marks >= 90:
        student[student_key]['grade'] = 'A'
    elif marks >= 80:
        student[student_key]['grade'] = 'B'
    elif marks >= 60:
        student[student_key]['grade'] = 'C'
    elif marks >= 40:
        student[student_key]['grade'] = 'D'
    else:
        student[student_key]['grade'] = 'Fail'

print("\nRecords of all students:")
for key, value in student.items():
    print(f"\n{key}: {value}")