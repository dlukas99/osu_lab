#imports
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, confusion_matrix , ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model



data_r = load_breast_cancer()

data = pd.DataFrame(data_r['data'], columns=data_r['feature_names'])
data['target'] = data_r['target']
# 0 -> maligant
# 1 -> benign

#1. zadatak
#a
print("Broj uzoraka: ", data.shape[0])
print("Broj znacajki: ", data_r['data'].shape[1])
print("Nazivi znacajki: ", data_r['feature_names'])
print("Nazivi izlaznih velicina: ", data_r['target_names'])
print("Vrijednosti izalznih velicina: ", data['target'].value_counts())

#b
plt.figure(figsize=(15, 12))
correlation_matrix = data.iloc[:, :-1].corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Korelacijska matrica ulaznih značajki')
plt.tight_layout()
plt.show()


#c
plt.figure(figsize=(20, 12))
plt.boxplot(data.iloc[:, :-1], tick_labels=data.columns[:-1], vert=False)
plt.title('Kutijasti dijagrami ulaznih značajki - prije normalizacije', fontsize=16)
plt.xlabel('Vrijednost', fontsize=14)
plt.ylabel('Značajka', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#e
scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data.iloc[:, :-1]),
    columns=data.columns[:-1]
)
data_scaled['target'] = data['target']

plt.figure(figsize=(20, 12))
plt.boxplot(data_scaled.iloc[:, :-1], tick_labels=data_scaled.columns[:-1], vert=False)
plt.title('Kutijasti dijagrami ulaznih značajki - nakon normalizacije', fontsize=16)
plt.xlabel('Standardizirana vrijednost', fontsize=14)
plt.ylabel('Značajka', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



#2. zadatak

data = load_breast_cancer()
X = data.data
y = data.target

X_df = pd.DataFrame(X, columns=data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#a
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

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
                    label=cl)
        


# Smanjenje dimenzionalnosti zbog vizualizacije
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model_pca = LogisticRegression(max_iter=1000, random_state=42)
model_pca.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, model_pca)
plt.title('Granica odluke logisticke regresije na skupu za učenje')
plt.show()

plot_decision_regions(X_test_pca, y_test, model_pca)
plt.title('Granica odluke logisticke regresije na testnom skupu')
plt.show()


# b
y_train_pred = model_pca.predict(X_train_pca)
y_test_pred = model_pca.predict(X_test_pca)

# c
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print(f"Metrike na skupu za ucenje:")
print(f"Tocnost: {train_accuracy:.4f}")
print(f"Preciznost: {train_precision:.4f}")
print(f"Odziv: {train_recall:.4f}")

print(f"\nMetrike na skupu za testiranje:")
print(f"Tocnost: {test_accuracy:.4f}")
print(f"Preciznost: {test_precision:.4f}")
print(f"Odziv: {test_recall:.4f}")

# d
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_pred))
disp.plot()
plt.title("Matrica zabune logisticke regresije")
plt.show()


#3. zadatak

data = load_breast_cancer()
X = data.data
y = data.target

X_df = pd.DataFrame(X, columns=data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# a
model = keras.Sequential()
model.add(layers.Input(shape=(X_train.shape[1], )))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

# b
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy",]
)


# c
history = model.fit(
    X_train_scaled,
    y_train,
    batch_size=16,
    epochs=50,
    validation_split = 0.1
)

# d
model.save('model.keras')

# e
model = load_model('model.keras')

score = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Loss: ", score[0])
print("Tocnost: ", score[1])

# f
predictions_prob = model.predict(X_test_scaled)
predictions_binary = (predictions_prob > 0.5).astype(int)

predictions_binary = predictions_binary.flatten()

cm = confusion_matrix(y_test, predictions_binary)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Matrica zabune neuronske mreže")
plt.show()
