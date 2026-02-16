class CPlusPlus:
    def inheritance(self):
        print("Multiple Inheritance")

    def polymorphism(self):
        print("Method OverLoading")
        print("Method OverRiding")


class Java:
    def inheritance(self):
        print("No Support Multiple Inheritance")

    def polymorphism(self):
        print("Method OverLoading")
        print("Method OverRiding")
        print("Constructor OverLoading")


def language(object_language):
    object_language.inheritance()
    object_language.polymorphism()


obj_c_plus_plus = CPlusPlus()
obj_java = Java()

language(obj_c_plus_plus)
language(obj_java)






