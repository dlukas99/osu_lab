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
'''
Zadatak 4.5.1 Skripta zadatak_1.py uˇcitava podatkovni skup iz data_C02_emission.csv.
Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih
numeriˇckih ulaznih veliˇcina. Detalje oko ovog podatkovnog skupa mogu se prona´ci u 3.
laboratorijskoj vježbi.
a) Odaberite željene numeriˇcke veliˇcine specificiranjem liste s nazivima stupaca. Podijelite
podatke na skup za uˇcenje i skup za testiranje u omjeru 80%-20%.
b) Pomo´cu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova
o jednoj numeriˇckoj veliˇcini. Pri tome podatke koji pripadaju skupu za uˇcenje oznaˇcite
plavom bojom, a podatke koji pripadaju skupu za testiranje oznaˇcite crvenom bojom.
c) Izvršite standardizaciju ulaznih veliˇcina skupa za uˇcenje. Prikažite histogram vrijednosti
jedne ulazne veliˇcine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja
transformirajte ulazne veliˇcine skupa podataka za testiranje.
d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
povežite ih s izrazom 4.6.
e) Izvršite procjenu izlazne veliˇcine na temelju ulaznih veliˇcina skupa za testiranje. Prikažite
pomoc´u dijagrama raspršenja odnos izmed¯u stvarnih vrijednosti izlazne velicˇine i procjene
dobivene modelom.
f) Izvršite vrednovanje modela na naˇcin da izraˇcunate vrijednosti regresijskih metrika na
skupu podataka za testiranje.
g) Što se dogad¯a s vrijednostima
'''
# Učitavanje podataka iz CSV datoteke
data = pd.read_csv('data_C02_emission.csv')

# a) Odabir numeričkih značajki (ulaznih varijabli)
numerical_features = ['Engine Size (L)', 'Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
x = data[numerical_features].to_numpy()  # Ulazne varijable kao NumPy niz
y = data['CO2 Emissions (g/km)'].to_numpy()  # Izlazna varijabla (ciljna)

# Podjela podataka na skup za učenje (80%) i testiranje (20%)
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state =1)

# b) Scatter dijagram: odnos između potrošnje goriva u gradu i emisije CO2
plt.figure()
plt.scatter(x_train[:,2], y_train, color='blue')  # Trening podaci
plt.scatter(x_test[:,2], y_test, color='red')     # Test podaci
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Odnos potrosnje i emisije")
plt.show()

# c) Histogram vrijednosti prije skaliranja (za jednu ulaznu varijablu)
plt.figure()
plt.hist(x_train[:, 2], color='blue')
plt.title("Prije skaliranja")
plt.show()

# Normalizacija (skaliranje) ulaznih podataka pomoću MinMax skalera
sc = MinMaxScaler()
X_train_n = sc.fit_transform(x_train)  # Skaliranje trening skupa
X_test_n = sc.transform(x_test)        # Primjena istog skalera na test skup

# Histogram vrijednosti nakon skaliranja
plt.figure()
plt.hist(X_train_n[:, 2], color='red')
plt.title("Poslije skaliranja")
plt.show()

# d) Treniranje linearnog regresijskog modela
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)  # Učenje modela
print(f"Koeficjenti: {linearModel.coef_}")        # Težine značajki
print(f"Intercept: {linearModel.intercept_}")     # Presjek s y-osom

# e) Predikcija na testnom skupu
y_test_p = linearModel.predict(X_test_n)

# Scatter dijagram stvarnih vs. predikcijskih vrijednosti
plt.figure()
plt.scatter(y_test, y_test_p)
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predikcijske vrijednosti")
plt.title("Odnos stvarnih vrijednosti izlazne veličine i procjene")
plt.show()

# f) Izračun regresijskih metrika za evaluaciju modela
MSE = mean_squared_error(y_test, y_test_p)                   # Srednja kvadratna pogreška
RMSE = np.sqrt(MSE)                                          # Korijen srednje kvadratne pogreške
MAE = mean_absolute_error(y_test, y_test_p)                  # Srednja apsolutna pogreška
MAPE = mean_absolute_percentage_error(y_test, y_test_p)      # Srednja apsolutna postotna pogreška
r2 = r2_score(y_test, y_test_p)                              # Koeficijent determinacije (R^2)

# Ispis evaluacijskih metrika
print(f"MSE:{MSE}\nRMSE:{RMSE}\nMAE:{MAE}\nMAPE:{MAPE}\nR2 Score:{r2}")
