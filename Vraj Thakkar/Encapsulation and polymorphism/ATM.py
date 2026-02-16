from bank import Axis, HDFC

print("Welcome to the ATM")

bank = input("Enter your bank (A for Axis / H for HDFC): ")

if bank.lower() == "a":
    print("You selected Axis Bank")
    a = Axis()
    a.withdraw()

elif bank.lower() == "h":
    print("You selected HDFC Bank")
    h = HDFC()
    h.withdraw()

else:
    print("No such bank provides service here.")
