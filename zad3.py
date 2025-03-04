#3
numbers=[]
while(True):
    try:
        num=input('Input number:')
        if num=='Done':
            break
        numbers.append(float(num))
    except:
        print("Not a number")
print(f"Number inputed: {len(numbers)}")
print(f"Average value: {sum(numbers)/len(numbers)}")
print(f"Minimal value: {min(numbers)}")
print(f"Maximal value: {max(numbers)}")
