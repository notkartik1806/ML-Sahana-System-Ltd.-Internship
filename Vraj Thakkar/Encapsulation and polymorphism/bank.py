class Axis:
    def __init__(self):
        self.maxamt = 30000
        self.txn_no = 0
        self.maxtxn = 3

    def withdraw(self):
        while self.txn_no < self.maxtxn:
            amount = int(input("How much do you want to withdraw: "))

            if amount <= self.maxamt:
                print(f"Transaction successful for ₹{amount}")
                self.txn_no += 1
                print(f"Transactions used: {self.txn_no}/{self.maxtxn}")
            else:
                print("Amount exceeds limit")

            choice = input("Do you want to continue? y/n: ")

            if choice.lower() != "y":
                break


class HDFC:
    def __init__(self):
        self.maxamt = 20000
        self.txn_no = 0
        self.maxtxn = 3

    def withdraw(self):
        while self.txn_no < self.maxtxn:
            amount = int(input("How much do you want to withdraw: "))

            if amount <= self.maxamt:
                print(f"Transaction successful for ₹{amount}")
                self.txn_no += 1
                print(f"Transactions used: {self.txn_no}/{self.maxtxn}")
            else:
                print("Amount exceeds limit")

            choice = input("Do you want to continue? y/n: ")

            if choice.lower() != "y":
                break
