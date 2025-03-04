#1
def total_euro(work_hours, pay):
    return work_hours*pay

work_hours=input('Radni sati: ')
work_hours=int(work_hours)

pay=input('eura/h: ')
pay=float(pay)


print(f"Ukupno: {total_euro(work_hours, pay)}")



