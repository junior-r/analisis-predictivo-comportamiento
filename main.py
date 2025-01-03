# Proyecto: Análisis Predictivo de Comportamiento de Clientes

## 1. Importar Librerías
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

## 2. Cargar y Preprocesar Datos
# Cargar el dataset
# Agregar el parámetro low_memory=False para evitar la advertencia de tipos mixtos
data = pd.read_csv('Amazon_Sale_Report.csv', low_memory=False)

# Eliminar columnas irrelevantes
columns_to_drop = ['Unnamed: 22', 'fulfilled-by', 'ship-country', 'currency', 'promotion-ids']
data_cleaned = data.drop(columns=columns_to_drop)

# Convertir fechas a formato datetime
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%m-%d-%y')

# Imputar valores faltantes
# Usar asignación directa en lugar de inplace para evitar advertencias
categorical_columns = ['ship-city', 'ship-state', 'Courier Status']
for col in categorical_columns:
    mode_value = data_cleaned[col].mode()[0]
    data_cleaned[col] = data_cleaned[col].fillna(mode_value)

median_amount = data_cleaned['Amount'].median()
data_cleaned['Amount'] = data_cleaned['Amount'].fillna(median_amount)

## 3. Análisis Descriptivo
# Visualizar distribuciones
def plot_histogram(column, title, xlabel):
    plt.figure(figsize=(8, 5))
    plt.hist(data_cleaned[column], bins=50, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frecuencia')
    plt.show()

plot_histogram('Amount', 'Distribución de Monto de Compra', 'Monto (INR)')

# Gráfico de Categorías
data_cleaned['Category'].value_counts().plot(kind='bar', alpha=0.7, edgecolor='black')
plt.title('Frecuencia de Categorías de Productos')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.show()

## 4. Segmentación de Clientes
# Selección de variables y normalización
segmentation_data = data_cleaned[['Amount', 'Qty']]
scaler = StandardScaler()
segmentation_data_scaled = scaler.fit_transform(segmentation_data)

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42)
data_cleaned['Cluster'] = kmeans.fit_predict(segmentation_data_scaled)

# Visualización de los clusters
plt.figure(figsize=(8, 6))
plt.scatter(segmentation_data_scaled[:, 0], segmentation_data_scaled[:, 1], c=data_cleaned['Cluster'], cmap='viridis', alpha=0.6, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroides')
plt.title('Segmentación de Clientes (K-means)')
plt.xlabel('Monto Normalizado (Amount)')
plt.ylabel('Cantidad Normalizada (Qty)')
plt.legend()
plt.show()

## 5. Desarrollo del Modelo Predictivo
# Crear variable objetivo
data_cleaned['Target'] = (data_cleaned['Qty'] > 0).astype(int)

# Dividir datos en entrenamiento y prueba
features = ['Amount', 'Qty']
X = data_cleaned[features]
y = data_cleaned['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predicciones
y_pred = rf_model.predict(X_test)

# Evaluación
print('Informe de Clasificación:\n', classification_report(y_test, y_pred))
print('Matriz de Confusión:\n', confusion_matrix(y_test, y_pred))

# Validación cruzada
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print('Precisión promedio de validación cruzada:', cv_scores.mean())

## 6. Importancia de Características
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
print('Importancia de Características:\n', importance_df)

## 7. Recomendaciones
# Basadas en los clusters y el modelo, las estrategias ya han sido redactadas en el informe final.
