#2
try:
    num=float(input('Input number:'))
    if num<0 or num>1: raise ValueError("Number not in interval")
    if num<0.6:
        print("F")
    elif num<0.7:
        print("D")
    elif num<0.8:
        print("C")
    elif num<0.9:
        print("B")
    else:
        print("A")
except ValueError as err:
    print(err)
