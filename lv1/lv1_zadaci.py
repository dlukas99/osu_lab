#1
def total_euro(work_hours, pay):
    return work_hours*pay

work_hours=input('Radni sati: ')
work_hours=int(work_hours)

pay=input('eura/h: ')
pay=float(pay)


print(f"Ukupno {total_euro(work_hours, pay)}")

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

#3
    
