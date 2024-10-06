# Importar librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar el archivo completo BI_Postulantes.xlsx
df = pd.read_excel('BI_Postulantes09-1.xlsx')

# Visualizar las primeras filas del archivo para verificar que los datos se cargaron correctamente
print(df.head())

# Preprocesamiento de datos
# Seleccionar las columnas numéricas para el análisis (ajusta según las columnas presentes en el archivo)
X = df[['Apertura Nuevos Conoc.', 'Nivel Organización', 'Participación Grupo Social', 
        'Grado Empatía', 'Grado Nerviosismo', 'Dependencia Internet']]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar el algoritmo k-means (con 3 clusters, puedes ajustar según el análisis)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Mostrar los conglomerados formados para cada postulante
print(df[['Postulante', 'Nom_Especialidad', 'Cluster']])

# Generar histogramas cruzando las dimensiones con las especialidades
for dim in ['Apertura Nuevos Conoc.', 'Nivel Organización', 'Participación Grupo Social',
            'Grado Empatía', 'Grado Nerviosismo', 'Dependencia Internet']:
    plt.figure(figsize=(10, 6))
    for especialidad in df['Nom_Especialidad'].unique():
        esp_data = df[df['Nom_Especialidad'] == especialidad]
        plt.hist(esp_data[dim], bins=10, alpha=0.5, label=f'Especialidad {especialidad}')
    plt.title(f'Histograma de {dim} por Especialidad')
    plt.xlabel(dim)
    plt.ylabel('Cantidad de postulantes')
    plt.legend()
    plt.show()
