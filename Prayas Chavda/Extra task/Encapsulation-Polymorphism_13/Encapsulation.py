# __ -> can't access variable outside the class via it's object directly
# _  ->it's use for naming convention purpose for inheritance
#  (python name mangling)


class Product:
    def __init__(self):
        self.__price = 500

    def sell(self):
        print(self.__price)

    def set_max_price(self, price):
        if price > 0:
            self.__price = price


c = Product()
c.sell()
# 500

c.__price = 1500
c.sell()
# 500

c.set_max_price(1500)
c.sell()
# 1500
