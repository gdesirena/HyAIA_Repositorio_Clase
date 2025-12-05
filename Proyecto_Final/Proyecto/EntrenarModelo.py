import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Support Vector Machine (SVM)
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- ENTRENAMIENTO DEL MODELO ---

# Definimos las rutas del archivo de datos y el archivo del modelo ya entrenado.
INPUT_FILE = 'Proyecto_Final/Proyecto/imagenesnumeros_data.npz'  # Archivo de datos resultado de las imagenes
MODEL_FILE = 'Proyecto_Final/Proyecto/reconocimiento_digitos_svm.pkl' # Archivo que tendrá el modelo entrenado.

# Función para entrenar el model usando SVM (Support Vector Machine)
def Entrenar_Modelo():
    
    print("Iniciando carga y entrenamiento del modelo...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: El archivo de datos no se encontró.  '{INPUT_FILE}'.")
        print("Ejecutar primero: 'LecturaImagenes.py' para crear el conjunto de datos a utilizar en el entrenamiento.")
        return

    # Cargamos el conjunto de datos  (X=features, y=targets)
    with np.load(INPUT_FILE) as data:
        X = data['X']
        y = data['y']

    # Dividimos el conjunto de datos en conjuntos de entrenamiento y prueba
    # Usamos 80% para entrenar y 20% para probar
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño del conjunto de pruebas: {X_test.shape[0]} muestras")

    # Inicializamos y entrenar el modelo SVM
    print("Entrenando modelo SVM. Favor de esperar...")
    
    # Utilizamos un kernel RBF para clasificación de imágenes.
    model = SVC(kernel='rbf', gamma='scale', C=10) 
    model.fit(X_train, y_train)

    print("Entrenamiento completado.")

    # Evaluamos el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 50)
    print(f"Precisión del modelo en la prueba: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

    # Guardamos el modelo entrenado
    joblib.dump(model, MODEL_FILE)
    print(f"El modelo fue guardado exitosamente en '{MODEL_FILE}'")

if __name__ == "__main__":
    import os
    Entrenar_Modelo()