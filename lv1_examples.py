#pr 1.1
x = 23
print ( x )
x = x + 7
print ( x ) # komentar : ispis varijable na ekranu

#pr 1.2
x = 23
y = x > 10
print ( y )

#pr 1.3
x = 23
if x < 10:
    print (" x je manji od 10 ")
else :
    print (" x je veci ili jednak od 10 ")  

#pr 1.4    
i = 5
while i > 0:
    print ( i )
    i = i - 1
print (" Petlja gotova ")
for i in range (0 , 5 ):
    print ( i )

#pr 1.5
lstEmpty = [ ]
lstFriend = ['Marko', 'Luka', 'Pero']
lstFriend.append ('Ivan')
print (lstFriend [0])
print (lstFriend [0:1:2])
print (lstFriend [ :2])
print (lstFriend [1: ])
print (lstFriend [1:3])
a = [1 , 2 , 3]
b = [4 , 5 , 6]
c = a + b
print ( c )
print ( max ( c) )
c[0] = 7
c.pop ()
for number in c:
    print ('List number', number )
print ('Done !')