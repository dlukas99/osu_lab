import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering
'''Zadatak 7.5.1 Skripta zadatak_1.py sadrži funkciju generate_data koja služi za generiranje
umjetnih podatkovnih primjera kako bi se demonstriralo grupiranje. Funkcija prima cijeli broj
koji definira željeni broju uzoraka u skupu i cijeli broj od 1 do 5 koji definira na koji naˇcin ´ce
se generirati podaci, a vra´ca generirani skup podataka u obliku numpy polja pri ˇcemu su prvi i
drugi stupac vrijednosti prve odnosno druge ulazne veliˇcine za svaki podatak. Skripta generira
500 podatkovnih primjera i prikazuje ih u obliku dijagrama raspršenja.
1. Pokrenite skriptu. Prepoznajete li koliko ima grupa u generiranim podacima? Mijenjajte
naˇcin generiranja podataka.
2. Primijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer
obojite ovisno o njegovoj pripadnosti pojedinoj grupi. Nekoliko puta pokrenite programski
kod. Mijenjate broj K. Što primje´cujete?
3. Mijenjajte naˇcin definiranja umjetnih primjera te promatrajte rezultate grupiranja podataka
(koristite optimalni broj grupa). Kako komentirate dobivene rezultate?'''

def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)  

kmeans = KMeans(n_clusters=4, random_state=0)  
y_kmeans = kmeans.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K-Means grupiranje')
plt.show()
