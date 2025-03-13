import numpy as np
import matplotlib.pyplot as plt

zeros = np.zeros((50, 50))
ones = np.ones((50, 50))

upper = np.hstack((zeros, ones))
lower = np.hstack((ones, zeros))
image = np.vstack((upper, lower))

plt.figure()
plt.imshow(image, cmap="gray")
plt.show()