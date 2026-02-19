# Manual Data student data Dictionary Created 

student = {
    's1': {
        'rollno': input("Enter roll number for s1: "),
        'name': input("Enter name for s1: "),
        'marks': int(input("Enter marks for s1: ")),
        'grade': None,
    },
    's2': {
        'rollno': input("Enter roll number for s2: "),
        'name': input("Enter name for s2: "),
        'marks': int(input("Enter marks for s2: ")),
        'grade': None,
    },
    's3': {
        'rollno': input("Enter roll number for s3: "),
        'name': input("Enter name for s3: "),
        'marks': int(input("Enter marks for s3: ")),
        'grade': None,
    },
    's4': {
        'rollno': input("Enter roll number for s4: "),
        'name': input("Enter name for s4: "),
        'marks': int(input("Enter marks for s4: ")),
        'grade': None,
    },
    's5': {
        'rollno': input("Enter roll number for s5: "),
        'name': input("Enter name for s5: "),
        'marks': int(input("Enter marks for s5: ")),
        'grade': None,
    }
}

# Assigning grades based on marks using nested for loop

for student_id in student:
    for key, value in student[student_id].items():
        if key == 'marks':
            marks = value
            if  marks <= 100 and marks >= 90:
                student[student_id]['grade'] = 'A'
            elif marks < 90 and marks >= 80:
                student[student_id]['grade'] = 'B'
            elif marks < 80 and marks >= 60:
                student[student_id]['grade'] = 'C'
            elif marks < 60 and marks >= 40:
                student[student_id]['grade'] = 'D'
            else:
                student[student_id]['grade'] = 'Fail'

# Displaying student dictionary with Grades Assigned

print("\n----------- Student Records -----------")
for student_id, details in student.items():
    print(f"\n{student_id}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
