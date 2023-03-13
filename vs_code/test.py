class alloMonCoco():

    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c 


class craque(alloMonCoco):

    def __str__(self):
        return str(f"{self.a}___{self.b}____{self.c}")


test = craque(1,2,3)
print(test)