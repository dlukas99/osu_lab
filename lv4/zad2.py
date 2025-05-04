import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder 

'''
Zadatak 4.5.2 Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoriˇcku
varijable „Fuel Type“ kao ulaznu veliˇcinu. Pri tome koristite 1-od-K kodiranje kategoriˇckih
veliˇcina. Radi jednostavnosti nemojte skalirati ulazne veliˇcine. Komentirajte dobivene rezultate.
Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
vozila radi?
'''

# Učitavanje podataka iz CSV datoteke
data = pd.read_csv('data_C02_emission.csv')

# Kreiranje OneHotEncoder objekta za kodiranje varijable 'Fuel Type'
ohe = OneHotEncoder()

# Transformacija 'Fuel Type' u binarni oblik i pretvaranje u DataFrame
ohe_df = pd.DataFrame(ohe.fit_transform(data[['Fuel Type']]).toarray())

# Spajanje kodiranih stupaca natrag na originalni DataFrame
data = data.join(ohe_df)

# Ručno postavljanje naziva svih stupaca (zbog novododanih kodiranih varijabli)
data.columns = ['Make','Model','Vehicle Class','Engine Size (L)','Cylinders','Transmission','Fuel Type',
                'Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)',
                'Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','CO2 Emissions (g/km)',
                'Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']  # 'Fuel0' do 'Fuel3' su kodirane vrijednosti goriva

# Odvajanje ciljne varijable (emisije CO2)
y = data['CO2 Emissions (g/km)'].copy()

# Uklanjanje ciljne varijable iz skupa podataka za X (ulazne značajke)
X = data.drop('CO2 Emissions (g/km)', axis=1)

# Podjela podataka na skup za treniranje i testiranje
X_train_all, X_test_all, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Odabir samo numeričkih i kodiranih značajki koje želimo koristiti za model
X_train = X_train_all[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)',
                       'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)',
                       'Fuel Consumption Comb (mpg)', 'Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]

X_test = X_test_all[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)',
                     'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)',
                     'Fuel Consumption Comb (mpg)', 'Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]

# Kreiranje i treniranje linearnog regresijskog modela
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

# Predikcija emisije CO2 na testnom skupu
y_test_p = linearModel.predict(X_test)

# Prikaz raspršenog dijagrama za usporedbu stvarnih i predviđenih vrijednosti
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, color='blue')  # Stvarne vrijednosti
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test_p, color='red')  # Predviđene vrijednosti
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()  # Automatski neće prikazati legende jer scatter ne sadrži labele
plt.show()

# Izračun maksimalne apsolutne pogreške predikcije
max_Error = max_error(y_test, y_test_p)
print('Max pogreska:', max_Error)

# Pronalazak modela vozila kod kojeg je model napravio najveću pogrešku
print('Model s max pogreskom:', X_test_all[abs(y_test - y_test_p == max_Error)]['Model'].iloc[0])
