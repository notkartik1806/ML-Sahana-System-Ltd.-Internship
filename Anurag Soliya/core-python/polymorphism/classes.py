class HDFC_bank:
    def __init__(self, balance):
        self.amount_limit = balance
        self.trans_limit = 3
        self.trans_done = 1
    
    def withdraw(self, amount):
        if self.trans_done > self.trans_limit:
            raise ValueError("transaction limit reached")   
        else:
            if amount > self.amount_limit:
                raise ValueError("Amount not under limit")
            else:
                print("collect the amount")
                self.amount_limit -= amount
                print(f"remaining amount: {self.amount_limit}")
                self.trans_done += 1
    
    def deposit(self, amount):
        self.amount_limit += amount
        print(f"New Limit: {self.amount_limit}")

class Axis:
    def __init__(self, balance):
        self.amount_limit = balance
        self.trans_limit = 5
        self.trans_done = 1
    
    def withdraw(self, amount):
        if self.trans_done > self.trans_limit:
            raise ValueError("transaction limit reached")   
        else:
            if amount > self.amount_limit:
                raise ValueError("Amount not under limit")
            else:
                print("collect the amount")
                self.amount_limit -= amount
                print(f"remaining amount: {self.amount_limit}")
                self.trans_done += 1

    
    def deposit(self, amount):
        self.amount_limit += amount
        print(f"New Limit: {self.amount_limit}")    