# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Cargar el archivo Excel
df = pd.read_excel('BI_Clientes09-1.xlsx')

# Ver las primeras filas del archivo
print(df.head())

# Preprocesamiento: Convertir las variables categóricas en numéricas usando LabelEncoder si es necesario
labelencoder = LabelEncoder()

# Convertir las columnas categóricas relevantes (esto es solo un ejemplo, ajusta a tus necesidades)
df['MaritalStatus'] = labelencoder.fit_transform(df['MaritalStatus'])  # Ejemplo: S=0, M=1
df['Gender'] = labelencoder.fit_transform(df['Gender'])  # F=0, M=1
df['CommuteDistance'] = labelencoder.fit_transform(df['CommuteDistance'])  # 0-1 Miles=0, 2-5 Miles=1, etc.
df['Region'] = labelencoder.fit_transform(df['Region'])  # Convertir la región

# Definir las características (X) y la columna objetivo (y)
X = df[['YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome', 'MaritalStatus', 
        'Gender', 'CommuteDistance', 'Age']]  # Puedes ajustar las características que consideres relevantes
y = df['BikeBuyer']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el Árbol de Decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluar el modelo
accuracy = clf.score(X_test, y_test)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Visualizar el árbol de decisión
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.show()
