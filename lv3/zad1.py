def scorer(score=50):
    if score>=90 and score<=100:
        print("A")
    elif score<90 and score>=80:
        print("B")
    else:
        print("You got low grade")
        
score=int(input("Input your score"))
scorer(score)
