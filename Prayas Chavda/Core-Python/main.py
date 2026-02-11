from student import AddStudent, ViewStudent, SearchStudent, DeleteStudent, BulkAddStudent

class StudentManager:
    def main(self):
        while True:
            print("\n" + "="*40)
            print("   STUDENT MANAGEMENT SYSTEM")
            print("="*40)
            print("1. Add a student")
            print("2. Bulk add students")
            print("3. View student list")
            print("4. Search for a student")
            print("5. Delete a student")
            print("6. Exit")
            print("="*40)

            choice = input("Enter your choice (1-6): ")
            
            if choice == '1':
                print("\n--- Add Student ---")
                AddStudent()
            elif choice == '2':
                BulkAddStudent()
            elif choice == '3':
                print("\n--- View Students ---")
                ViewStudent()
            elif choice == '4':
                print("\n--- Search Student ---")
                SearchStudent()
            elif choice == '5':
                print("\n--- Delete Student ---")
                DeleteStudent()
            elif choice == '6':
                print("\nExiting the Student Manager. Adios!")
                break
            else:
                print("\nInvalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    manager = StudentManager()
    manager.main()