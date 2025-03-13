import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
#uk broj osoba
num_people = data.shape[0]
print(f"Number of people: {num_people}")

height = data[:, 1]
weight = data[:, 2]

#odnos visine i mase
plt.figure(figsize=(8, 6))
plt.scatter(height, weight, label='All data')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('odnos visine i maset')
plt.legend()
plt.show()

# svaka 50 osoba, odnos visine i mase
height_50 = height[::50]
weight_50 = weight[::50]

plt.figure(figsize=(8, 6))
plt.scatter(height_50, weight_50, label='Every 50th person', color='red')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('svaka 50 osoba, odnos visine i mase')
plt.legend()
plt.show()


min_height = np.min(height)
max_height = np.max(height)
average_height = np.mean(height)

print(f"Minimum height: {min_height} cm")
print(f"Maximum height: {max_height} cm")
print(f"Average height: {average_height} cm")

# za muskarce i zene
ind_men = (data[:, 0] == 1) 
ind_women = (data[:, 0] == 0)  

height_men = data[ind_men, 1]
weight_men = data[ind_men, 2]

height_women = data[ind_women, 1]
weight_women = data[ind_women, 2]

#za muske
min_height_men = np.min(height_men)
max_height_men = np.max(height_men)
average_height_men = np.mean(height_men)

# za zene
min_height_women = np.min(height_women)
max_height_women = np.max(height_women)
average_height_women = np.mean(height_women)

print(f"Muskarci:")
print(f"Minimum height: {min_height_men} cm")
print(f"Maximum height: {max_height_men} cm")
print(f"Average height: {average_height_men} cm")

print(f"Zene:")
print(f"Minimum height: {min_height_women} cm")
print(f"Maximum height: {max_height_women} cm")
print(f"Average height: {average_height_women} cm")
