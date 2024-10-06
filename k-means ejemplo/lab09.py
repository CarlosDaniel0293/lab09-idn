import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos desde un archivo Excel
# Asegúrate de que 'data.xlsx' esté en la misma carpeta o especifica la ruta completa
data = pd.read_excel('data.xlsx', engine='openpyxl')

# Mostrar las primeras filas del archivo
print(data.head())

# Seleccionar las columnas que se van a usar para el clustering (por ejemplo, 'Feature1' y 'Feature2')
X = data[['Feature1', 'Feature2']]

# Crear el modelo KMeans
# n_clusters es el número de clusters que queremos crear
kmeans = KMeans(n_clusters=3, random_state=0)

# Ajustar el modelo a los datos
kmeans.fit(X)

# Predecir los clusters para cada punto de datos
data['Cluster'] = kmeans.predict(X)

# Mostrar los datos con sus clusters
print(data)

# Visualizar los clusters
plt.scatter(data['Feature1'], data['Feature2'], c=data['Cluster'], cmap='viridis')

# Visualizar los centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroides')
plt.title('K-means Clustering')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()
