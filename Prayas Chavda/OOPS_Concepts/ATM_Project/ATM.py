from Bank import HDFC,AXIS

class ATM:
    
    def __init__(self):
        self.HDFC = HDFC()
        self.AXIS = AXIS()
        
        print("Welcome to the ATM!")
        print("Please select your bank:")
        print("1. AXIS")
        print("2. HDFC")
        
        choice = int(input("Enter your choice: "))
        
        if choice == 1:
            print("You have selected AXIS.")
            self.perform_transactions(self.AXIS, "AXIS")
            
        elif choice == 2:
            print("You have selected HDFC.")
            self.perform_transactions(self.HDFC, "HDFC")
            
        else:
            print("Invalid choice. Please try again.")
    
    def perform_transactions(self, bank, bank_name):
        while True:
            bank.check_balance()
            amount = int(input("Enter the amount to withdraw (or 0 to exit): "))
            
            if amount == 0:
                break
            
            bank.withdraw(amount)
            bank.check_balance()
            bank.transactions()
            
            continue_choice = input("Do you want to perform another transaction? (yes/no): ")
            if continue_choice.lower() != "yes":
                break


if __name__ == "__main__":
    ATM()