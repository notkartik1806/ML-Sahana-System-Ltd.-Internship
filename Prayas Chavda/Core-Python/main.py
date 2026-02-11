from student import StudentManager

if __name__ == "__main__":
    manager = StudentManager()
    
    count = int(input("Enter count of data entry: "))
    manager.get_data(count)
    
    sub_count = int(input("\nEnter subject count: "))
    manager.add_subjects(sub_count)
    
    manager.write_data()
    manager.display_all_students()
