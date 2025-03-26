import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

data=pd.read_csv('data_C02_emission.csv')
#a
numerical_features = ['Engine Size (L)', 'Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
x = data[numerical_features].to_numpy()
y = data['CO2 Emissions (g/km)'].to_numpy()
x_train , x_test , y_train , y_test = train_test_split (x , y , test_size = 0.2 , random_state =1 )

#b
plt.figure()
plt.scatter(x_train[:,2], y_train, color='blue')
plt.scatter(x_test[:,2], y_test, color='red')
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Odnos potrosnje i emisije")
plt.show()

#c
plt.figure()
plt.hist(x_train[:, 2], color='blue')
plt.title("Prije skaliranja")
plt.show()

sc = MinMaxScaler ()
X_train_n = sc.fit_transform ( x_train )
X_test_n = sc.transform ( x_test )
plt.figure()
plt.hist(X_train_n[:, 2], color='red')
plt.title("Poslije skaliranja")
plt.show()

#d
linearModel = lm.LinearRegression ()
linearModel.fit( X_train_n , y_train )
print(f"Koeficjenti: {linearModel.coef_}")
print(f"Intercept: {linearModel.intercept_}")

#e
y_test_p = linearModel.predict ( X_test_n )
plt.figure()
plt.scatter(y_test, y_test_p)
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predikcijske vrijednosti")
plt.title("Odnos stvarnih vrijednosti izlazne veličine i procjene")
plt.show()

#f
MSE= mean_squared_error(y_test , y_test_p )
RMSE=np.sqrt(MSE)
MAE = mean_absolute_error( y_test , y_test_p )
MAPE=mean_absolute_percentage_error(y_test, y_test_p)
r2=r2_score(y_test, y_test_p)
print(f"MSE:{MSE}\nRMSE:{RMSE}\nMAE:{MAE}\nMAPE:{MAPE}\nR2 Score:{r2}")