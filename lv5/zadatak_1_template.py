import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

'''Zadatak 5.5.1 Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije
ulazne veliˇcine. Podaci su podijeljeni na skup za uˇcenje i skup za testiranje modela.
a) Prikažite podatke za uˇcenje u x1−x2 ravnini matplotlib biblioteke pri ˇcemu podatke obojite
s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
cmap kojima je mogu´ce definirati boju svake klase.
b) Izgradite model logistiˇcke regresije pomo´cu scikit-learn biblioteke na temelju skupa podataka
za uˇcenje.
c) Prona ¯ dite u atributima izgra ¯ denog modela parametre modela. Prikažite granicu odluke
nauˇcenog modela u ravnini x1 −x2 zajedno s podacima za uˇcenje. Napomena: granica
odluke u ravnini x1−x2 definirana je kao krivulja: θ0+θ1x1+θ2x2 = 0.
d) Provedite klasifikaciju skupa podataka za testiranje pomoc´u izgrad¯enog modela logisticˇke
regresije. Izraˇcunajte i prikažite matricu zabune na testnim podacima. Izraˇcunate toˇcnost,
preciznost i odziv na skupu podataka za testiranje.
e) Prikažite skup za testiranje u ravnini x1−x2. Zelenom bojom oznaˇcite dobro klasificirane
primjere dok pogrešno klasificirane primjere oznaˇcite crnom bojom.'''

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


'''a) Prikažite podatke za ucenje u ˇ x1 −x2 ravnini matplotlib biblioteke pri cemu podatke obojite ˇ
s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
cmap kojima je moguce de ´ finirati boju svake klase.'''

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm',marker='o', alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm',marker='x', alpha=0.8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
'''b) Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa poda- ´
taka za ucenje.'''
LogRegression_model = LogisticRegression ()
LogRegression_model.fit( X_train , y_train )


'''c) Pronadite u atributima izgra ¯ denog modela parametre modela. Prikažite granicu odluke ¯
naucenog modela u ravnini ˇ x1 − x2 zajedno s podacima za ucenje. Napomena: granica ˇ
odluke u ravnini x1 −x2 definirana je kao krivulja: θ0 +θ1x1 +θ2x2 = 0.'''


theta1, theta2=LogRegression_model.coef_[0]  
theta0=LogRegression_model.intercept_[0] 

def decision_boundary(x1):
    return (-theta0 - theta1 * x1) / theta2

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', alpha=0.8)

x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
plt.plot(x_values, decision_boundary(x_values), color='black', label='Granica odluke')

plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


''''d) Provedite klasifikaciju skupa podataka za testiranje pomocu izgra ´ denog modela logisti ¯ cke ˇ
regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izra ˇ cunate to ˇ cnost, ˇ
preciznost i odziv na skupu podataka za testiranje.'''
# stvarna vrijednost izlazne velicine i predikcija
y_pred = LogRegression_model.predict(X_test)
# matrica zabune
cm=confusion_matrix(y_test, y_pred)
print (" Matrica zabune : " , cm )
disp = ConfusionMatrixDisplay ( confusion_matrix=cm )
disp.plot ()
plt.show ()
# tocnost
print (" Tocnost : " , accuracy_score ( y_test , y_pred ) )
#preciznost
print (" Preciznost : " , precision_score(y_test, y_pred) )
#odziv
print (" Odziv : " , recall_score(y_test, y_pred) )



'''e) Prikažite skup za testiranje u ravnini x1 −x2. Zelenom bojom oznacite dobro klasi ˇ ficirane
primjere dok pogrešno klasificirane primjere oznacite crnom bojom. '''

# Prikaz testnog skupa s oznakama za dobro i loše klasificirane primjere
correctly_classified = (y_test == y_pred)
misclassified = ~correctly_classified

plt.scatter(X_test[correctly_classified, 0], X_test[correctly_classified, 1], 
            color='green', marker='o', label='Correctly Classified', alpha=0.8)
plt.scatter(X_test[misclassified, 0], X_test[misclassified, 1], 
            color='black', marker='x', label='Misclassified', alpha=0.8)

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Test Set Classification Results')
plt.show()
