from student_manager import StudentManager

def main():
    manager = StudentManager()

    count = int(input("Enter number of students: "))
    manager.getdata(count)

    sub_count = int(input("Enter number of subjects: "))
    manager.add_sub(sub_count)

    manager.write_data()

    print("Data saved successfully!")

if __name__ == "__main__":
    main()
