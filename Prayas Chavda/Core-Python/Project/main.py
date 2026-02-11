class StudentManager:
    def main(self):
        print("Welcome to the Student Manager!")
        # Here you can add code to manage students, such as adding, removing, or displaying student information.
        print("Choose one of the following options:")
        print("1. Add a student")
        print("2. View a student")
        print("3. Search for a student")
        print("4. Delete a student")
        print("5. Exit")

        choice = input("Enter your choice: ")
        if choice == '1':
            print("Adding a student.")
        elif choice == '2':
            print("Viewing a student.")
        elif choice == '3':
            print("Searching for a student.")
        elif choice == '4':
            print("Deleting a student.")
        elif choice == '5':
            print("Exiting the Student Manager. Goodbye!")
        
        else:
            print("Invalid choice. Please try again.")

Manager = StudentManager()
Manager.main()