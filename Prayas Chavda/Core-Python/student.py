class AddStudent:
    def __init__(self):
        self.name = input("Enter student name: ")
        self.age = input("Enter student age: ")
        self.marks = input("Enter student marks: ")
        self.save_to_file()

    def save_to_file(self):
        with open("data.txt", "a") as file:
            file.write(f"{self.name},{self.age},{self.marks}\n")
        print("Student added successfully!")

class BulkAddStudent:
    def __init__(self):
        self.students = []
        self.add_bulk()
    
    def add_bulk(self):
        print("\n--- Bulk Add Students ---")
        print("\nAdd multiple students manually")
        
        self.manual_bulk_add()
    
    def manual_bulk_add(self):
        try:
            num_students = int(input("\nHow many students do you want to add? "))
            if num_students <= 0:
                print("Please enter a positive number!")
                return
            
            print(f"\nYou will now enter details for {num_students} students:\n")
            
            for i in range(num_students):
                print(f"--- Student {i+1} ---")
                name = input(f"Enter student name: ")
                age = input(f"Enter student age: ")
                marks = input(f"Enter student marks: ")
                self.students.append((name, age, marks))
                print()
            
            self.save_all_to_file()
            
        except ValueError:
            print("Invalid input! Please enter a valid number.")
    
    def save_all_to_file(self):
        try:
            with open("data.txt", "a") as file:
                for student in self.students:
                    file.write(f"{student[0]},{student[1]},{student[2]}\n")
            print(f"\nSuccessfully added {len(self.students)} student(s)!")
        except Exception as e:
            print(f"Error saving students: {e}")

class ViewStudent:
    def __init__(self):
        self.view_all()
    
    def view_all(self):
        try:
            with open("data.txt", "r") as file:
                students = file.readlines()
                if not students:
                    print("No students found!")
                else:
                    print("\n--- Student List ---")
                    for i, student in enumerate(students, 1):
                        data = student.strip().split(",")
                        if len(data) == 3:
                            print(f"{i}. Name: {data[0]}, Age: {data[1]}, Marks: {data[2]}")

        except FileNotFoundError:
            print("No students found!")

class SearchStudent:
    def __init__(self):
        self.search_name = input("Enter student name to search: ")
        self.search()
    
    def search(self):
        found = False
        try:
            with open("data.txt", "r") as file:
                students = file.readlines()
                for student in students:
                    data = student.strip().split(",")
                    if len(data) == 3 and data[0].lower() == self.search_name.lower():
                        print(f"\nStudent Found!")
                        print(f"Name: {data[0]}, Age: {data[1]}, Marks: {data[2]}")
                        found = True
                        break
                if not found:
                    print("Student not found!")
        except FileNotFoundError:
            print("Student not found!")

class DeleteStudent:
    def __init__(self):
        self.delete_name = input("Enter student name to delete: ")
        self.delete()
    
    def delete(self):
        found = False
        try:
            with open("data.txt", "r") as file:
                students = file.readlines()
            
            with open("data.txt", "w") as file:
                for student in students:
                    data = student.strip().split(",")
                    if len(data) == 3 and data[0].lower() == self.delete_name.lower():
                        found = True
                        print(f"Student '{data[0]}' deleted successfully!")
                    else:
                        file.write(student)
            
            if not found:
                print("Student not found!")
        except FileNotFoundError:
            print("No students found!")