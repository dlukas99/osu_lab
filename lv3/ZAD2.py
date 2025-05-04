'''
Zadatak 3.4.2 Napišite programski kod koji ce prikazati sljede ´ ce vizualizacije: ´
a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz. ´
b) Pomocu dijagrama raspršenja prikažite odnos izme ´ du gradske potrošnje goriva i emisije ¯
C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu¯
velicina, obojite to ˇ ckice na dijagramu raspršenja s obzirom na tip goriva. ˇ
c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip ´
goriva. Primjecujete li grubu mjernu pogrešku u podacima? ´
d) Pomocu stup ´ castog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
groupby.
e) Pomocu stup ´ castog grafa prikažite na istoj slici prosje ˇ cnu C02 emisiju vozila s obzirom na ˇ
broj cilindara.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

#a)
plt.hist(data['CO2 Emissions (g/km)'], bins=10, color='blue', edgecolor='black')
plt.xlabel("CO2 Emissions (g/km)")
plt.title("Distribucija emisije CO2")
plt.show()

#b)
data["Fuel Type"] = data["Fuel Type"].astype("category")

data.plot.scatter(x="Fuel Consumption City (L/100km)",
            y="CO2 Emissions (g/km)",
            c="Fuel Type", cmap="cool", s=20)
plt.title("Odnos gradske potrošnje goriva i emisije CO2 plinova")
plt.show()

#c)
data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.xlabel("Tip goriva")
plt.ylabel("Izvangradska potrošnja goriva (L/100km)")
plt.title("Izvangradska potrošnja goriva po tipu goriva")
plt.suptitle("")
plt.show()

#d)
data.groupby('Fuel Type').size().plot(kind='bar', color='cyan', edgecolor='black')
plt.xlabel("Tip goriva")
plt.ylabel("Broj vozila")
plt.title("Broj vozila po tipu goriva")
plt.show()

#e)
data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().plot(kind='bar', color='lightblue', edgecolor='black')
plt.xlabel("Broj cilindara")
plt.ylabel("Prosječna emisija CO2 (g/km)")
plt.title("Prosječna emisija CO2 s obzirom na broj cilindara")
plt.show()
