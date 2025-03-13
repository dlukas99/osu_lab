import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,3,3,2,1])
y=np.array([1,1,2,2,1])
plt.plot (x , y , 'b', linewidth =1 , marker =".", markersize =10 )
plt.axis ([0 ,4.0,0,4.0])
plt.xlabel ('x')
plt.ylabel ('y')
plt.title ('slika')
plt.show ()
