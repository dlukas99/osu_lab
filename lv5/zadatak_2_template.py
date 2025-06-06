import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


'''Zadatak 5.5.2 Skripta zadatak_2.py uˇcitava podatkovni skup Palmer Penguins [1]. Ovaj
podatkovni skup sadrži mjerenja provedena na tri razliˇcite vrste pingvina (’Adelie’, ’Chinstrap’,
’Gentoo’) na tri razliˇcita otoka u podruˇcju Palmer Station, Antarktika. Vrsta pingvina
odabrana je kao izlazna veliˇcina i pri tome su klase oznaˇcene s cjelobrojnim vrijednostima
0, 1 i 2. Ulazne veliˇcine su duljina kljuna (’bill_length_mm’) i duljina peraje u mm (’flipper_
length_mm’). Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna
funkcija plot_decision_region.
a) Pomo´cu stupˇcastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu
pingvina) u skupu podataka za uˇcenje i skupu podataka za testiranje. Koristite numpy
funkciju unique.
b) Izgradite model logistiˇcke regresije pomo´cu scikit-learn biblioteke na temelju skupa podataka
za uˇcenje.
c) Pronad¯ite u atributima izgrad¯enog modela parametre modela. Koja je razlika u odnosu na
binarni klasifikacijski problem iz prvog zadatka?
d) Pozovite funkciju plot_decision_region pri ˇcemu joj predajte podatke za uˇcenje i
izgrad¯eni model logisticˇke regresije. Kako komentirate dobivene rezultate?
e) Provedite klasifikaciju skupa podataka za testiranje pomoc´u izgrad¯enog modela logisticˇke
regresije. Izraˇcunajte i prikažite matricu zabune na testnim podacima. Izraˇcunajte toˇcnost.
Pomo´cu classification_report funkcije izraˇcunajte vrijednost ˇcetiri glavne metrike'''

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

'''a) Pomocu stup ´ castog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu ˇ
pingvina) u skupu podataka za ucenje i skupu podataka za testiranje. Koristite numpy ˇ
funkciju unique.'''
classes, counts_train=np.unique(y_train, return_counts=True)
classes, counts_test=np.unique(y_test, return_counts=True)
X_axis = np.arange(len(classes))
plt.bar(X_axis - 0.2, counts_train, 0.4, label = 'Train')
plt.bar(X_axis + 0.2, counts_test, 0.4, label = 'Test') 
plt.xticks(X_axis, ['Adelie(0)', 'Chinstrap(1)', 'Gentoo(2)'])
plt.xlabel("Penguins")
plt.ylabel("Counts")
plt.legend()
plt.show()

'''
b) Izgradite model logisticke regresije pomo ˇ cu scikit-learn biblioteke na temelju skupa poda- ´
taka za ucenje. '''
logisticRegression = LogisticRegression(max_iter=120)
logisticRegression.fit(X_train,y_train.ravel())

'''
c) Pronadite u atributima izgra ¯ denog modela parametre modela. Koja je razlika u odnosu na ¯
binarni klasifikacijski problem iz prvog zadatka?'''
teta0 = logisticRegression.intercept_ 
coefs = logisticRegression.coef_ 
print('Teta0:')
print(teta0)
print('Parametri modela')
print(coefs)

'''

d) Pozovite funkciju plot_decision_region pri cemu joj predajte podatke za u ˇ cenje i ˇ
izgradeni model logisti ¯ cke regresije. Kako komentirate dobivene rezultate? ˇ'''
y_train = y_train.ravel()
y_test = y_test.ravel()

plot_decision_regions(X_train, y_train, logisticRegression)

'''
e) Provedite klasifikaciju skupa podataka za testiranje pomocu izgra ´ denog modela logisti ¯ cke ˇ
regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izra ˇ cunajte to ˇ cnost. ˇ
Pomocu c ´ lassification_report funkcije izracunajte vrijednost ˇ cetiri glavne metrike'''
y_prediction = logisticRegression.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test,y_prediction))
disp.plot()
plt.title('Matrica zabune')
plt.show()
print(f'Tocnost: {accuracy_score(y_test,y_prediction)}')
print(classification_report(y_test,y_prediction))
