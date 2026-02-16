class AXIS:
    def __init__(self):
        self.__max_transactions = 3
        self.__max_amount = 20000
        self.amount = 0
        self.amt = 23500
        self.__transaction_count = 0

    def withdraw(self, amount):
        if self.__transaction_count >= self.__max_transactions:
            print("Maximum transactions exceeded.")
            return
        
        if amount > self.__max_amount:
            print("You have exceeded the maximum amount limit.")
            return

        if amount > self.amt:
            print("Insufficient balance.")
        else:
            self.amount = amount
            self.amt -= amount
            self.__transaction_count += 1
            
    def check_balance(self):
        print("Your current balance is: ", self.amt)


class HDFC:
    def __init__(self):
        self.__max_transactions = 5
        self.__max_amount = 30000
        self.amount = 0
        self.amt = 78500
        self.__transaction_count = 0

    def withdraw(self, amount):
        if self.__transaction_count >= self.__max_transactions:
            print("Maximum transactions exceeded.")
            return
        
        if amount > self.__max_amount:
            print("You have exceeded the maximum amount limit.")
            return

        if amount > self.amt:
            print("Insufficient balance.")
        else:
            self.amount = amount
            self.amt -= amount
            self.__transaction_count += 1
    
    def check_balance(self):
        print("Your current balance is: ", self.amt)
    
    def transactions(self):
        print("You have done ", self.__transaction_count, " transaction(s) today.")