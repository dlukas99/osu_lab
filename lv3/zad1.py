'''
Zadatak 3.4.1 Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljede ´ ca pitanja: ´
a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili ˇ
duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke veli ˇ cine konvertirajte u tip ˇ
category.
b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: ´
ime proizvoda¯ ca, model vozila i kolika je gradska potrošnja. ˇ
c) Koliko vozila ima velicinu motora izme ˇ du 2.5 i 3.5 L? Kolika je prosje ¯ cna C02 emisija ˇ
plinova za ova vozila?
d) Koliko mjerenja se odnosi na vozila proizvoda¯ ca Audi? Kolika je prosje ˇ cna emisija C02 ˇ
plinova automobila proizvoda¯ ca Audi koji imaju 4 cilindara? ˇ
e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na ˇ
broj cilindara?
f) Kolika je prosjecna gradska potrošnja u slu ˇ caju vozila koja koriste dizel, a kolika za vozila ˇ
koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva? ´
h) Koliko ima vozila ima rucni tip mjenja ˇ ca (bez obzira na broj brzina)? ˇ
i) Izracunajte korelaciju izme ˇ du numeri ¯ ckih veli ˇ cina. Komentirajte dobiveni rezultat.
'''

import numpy as np
import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#a)
total_measurements = data.shape[0]
columns_info = data.dtypes
missing_values = data.isnull().sum().sum()
duplicated_values = data.duplicated().sum()

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

categorical_columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
data[categorical_columns] = data[categorical_columns].astype('category')

#b)
top_cars = data.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(3)
worst_cars = data.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).tail(3)

#c)
filtered_cars = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
num_filtered_cars = filtered_cars.shape[0]
avg_CO2 = filtered_cars['CO2 Emissions (g/km)'].mean()

#d)
audi_cars = data[data['Make'] == 'Audi']
audi_count = audi_cars.shape[0]
audi_4_avg_CO2 = audi_cars[audi_cars['Cylinders'] == 4]['CO2 Emissions (g/km)'].mean()

#e)
cylinders_analysis = data.groupby('Cylinders')['CO2 Emissions (g/km)'].agg(['count', 'mean'])

#f)
fuel_analysis = data.groupby('Fuel Type')['Fuel Consumption City (L/100km)'].agg(['mean', 'median'])

#g)
diesel_4_cars = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
worst_diesel_4 = diesel_4_cars.sort_values('Fuel Consumption City (L/100km)', ascending=False).head(1)

#h)
manual_cars_count = data[data['Transmission'].astype(str).str.startswith('M', na=False)].shape[0]

#i)
correlation = data.corr(numeric_only = True)

#a)
print("Ukupno mjerenja: ", total_measurements)
print("Tip svake veličine: \n", columns_info)
print("Broj izostalih vrijednosti: ", missing_values)
print("Broj dupliciranih vrijednosti: ", duplicated_values)

#b)
print("Tri vozila s najvećom potrošnjom:\n", top_cars[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
print("Tri vozila s najmanjom potrošnjom:\n", worst_cars[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

#c)
print(f"Broj vozila između 2.5 i 3.5 L: {num_filtered_cars}, Prosječna CO2 emisija: {avg_CO2}")

#d)
print(f"Broj Audi vozila: {audi_count}, Prosječna CO2 emisija za Audi s 4 cilindra: {audi_4_avg_CO2}")

#e)
print("Analiza cilindara: \n", cylinders_analysis)

#f)
print("Prosječna i medijalna gradska potrošnja: \n", fuel_analysis)

#g)
print("Najveća gradska potrošnja - 4 cilindra: \n", worst_diesel_4[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

#h)
print("Broj vozila s ručnim mjenjačem:", manual_cars_count)

#i)
print("Korelacija između numeričkih veličina:", correlation)
