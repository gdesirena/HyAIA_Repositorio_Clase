import cv2
import numpy as np
import os
from tqdm import tqdm  # Lo uso para mostrar avance en la carga de imagenes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Machine (SVM)
from sklearn.metrics import accuracy_score, classification_report
import joblib


class ML_Predict:
    def __init__(self, parImagePath, parImageOutputFile, parModelPath):
        self.ImagePath = parImagePath
        self.imageOutputFile = parImageOutputFile
        self.ModelPath = parModelPath
        self.IMG_SIZE = 28
        self.SVC_Model = None

    def load_and_process_images(self):
        # Se almacenarán las características (Todos los píxeles de cada dígito)
        data = []
        labels = []  # Se almacenarán las etiquetas (dígitos)

        # Obtenemos los nombres de las subcarpetas (los dígitos 0-9) y también lasusaremos como etiquetas ó targets del modelo
        digit_folders = sorted([d for d in os.listdir(
            self.ImagePath) if os.path.isdir(os.path.join(self.ImagePath, d))])

        print(f"\nCarpetas de dígitos encontradas: {digit_folders}")

        # Para cada uno de los folders procesamos las imagenes
        for digit in digit_folders:
            folder_path = os.path.join(self.ImagePath, digit)
            label = int(digit)  # La etiqueta es el nombre de la carpeta

            # Iteramos por cada imágen en la carpeta del dígito actual
            for filename in tqdm(os.listdir(folder_path), desc=f"Procesando dígito {digit}"):
                # Los tipos de imagenes soportados son: .png, .jpg y .jpeg.   Entre más cantidad de imagenes por cada dígito mejor accuracy lograremos tener.
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, filename)

                    try:
                        # Leemos la imagen en escala de grises para estandarizar el color
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                        if img is None:
                            print(
                                f"Advertencia: No se pudo cargar la imagen {img_path}")
                            continue

                        # Normalizamos el tamaño a IMG_SIZE x IMG_SIZE (28 X 28)
                        img_resized = cv2.resize(
                            img, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_AREA)

                        # Convertimos a Blanco y Negro para que el módelo funcione mejor con el fondo negro (0) y el dígito blanco (255)
                        _, img_binary = cv2.threshold(
                            img_resized, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

                        # Estandarizamos la imagen: de matriz 28x28 en vector 784
                        features = img_binary.flatten()

                        # 6. Normalizamos los valores de píxeles de 0-255 y dejar todo en 0 y 1
                        features = features / 255.0

                        data.append(features)
                        labels.append(label)

                    except Exception as e:
                        print(f"Error al procesar {img_path}: {e}")

        # Convertimos las listas a arrays de NumPy
        X = np.array(data)
        y = np.array(labels)

        print(f"\nTotales de datos procesados (X): {X.shape}")
        print(f"Total de etiquetas (y): {y.shape}")

        # Guardamos el conjuto de datos en un archivo NPZ para usarlo en el entrenamiento
        np.savez_compressed(self.imageOutputFile, X=X, y=y)
        print(
            f"Archivo de datos guardado exitosamente en {self.imageOutputFile}")

    def Entrenar_Modelo(self):
        if self.SVC_Model != None:
            return
        print("Iniciando carga y entrenamiento del modelo...")

        if not os.path.exists(self.imageOutputFile):
            print(
                f"Error: El archivo de datos no se encontró.  '{self.imageOutputFile}'.")
            print("Ejecutar primero: 'LecturaImagenes.py' para crear el conjunto de datos a utilizar en el entrenamiento.")
            return

        # Cargamos el conjunto de datos  (X=features, y=targets)
        with np.load(self.imageOutputFile) as data:
            X = data['X']
            y = data['y']

        # Dividimos el conjunto de datos en conjuntos de entrenamiento y prueba
        # Usamos 80% para entrenar y 20% para probar
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(
            f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
        print(f"Tamaño del conjunto de pruebas: {X_test.shape[0]} muestras")

        # Inicializamos y entrenar el modelo SVM
        print("Entrenando modelo SVM. Favor de esperar...")

        # Utilizamos un kernel RBF para clasificación de imágenes.
        model = SVC(kernel='rbf', gamma='scale', C=10, probability=True)
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
        joblib.dump(model,  self.ModelPath)
        self.SVC_Model = model
        print(f"El modelo fue guardado exitosamente en '{ self.ModelPath}'")
        return accuracy, y_pred

    def Procesar_Imagen(self, img_path):
        # Leemos la imagen en escala de grises
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(
                f"Error: No se pudo cargar la imagen desde la ruta especificada: {img_path}")

        # Normalizamos el tamaño a IMG_SIZE x IMG_SIZE (28 X 28)
        img_resized = cv2.resize(
            img, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Convertimos a blanco y negro (Invertido)
        _, img_binary = cv2.threshold(
            img_resized, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Aplanamos la imagen
        features = img_binary.flatten() / 255.0

        # Aseguramos que los datos sean en 2D para la predicción (1 fila, 784 columnas)
        return features.reshape(1, -1)

    def Predecir_Digito(self,  parImagePath):
        print("\n--- Reconocedor de Dígitos ---")

        if not os.path.exists(self.ModelPath):
            print(
                f"Error: No se encontró el modelo entrenado '{ self.ModelPath}'.")
            print("Asegúrate de ejecutar primero el archivo 'EntrenarModelo.py'.")
            return

        # 1. Cargar el modelo entrenado
        try:
            model = joblib.load(self.ModelPath)
        except Exception as e:
            print(f"Error: No fue posible cargar el modelo: {e}")
            return ""
        try:
            # Procesar la imagen
            features = self.Procesar_Imagen(parImagePath)

            # Realizar la predicción
            prediction = model.predict(features)
            y_prob = model.predict_proba(features)
            predicted_digit = prediction[0]

            digito = np.argmax(y_prob[0])
            confianza = y_prob[0][digito]
            porcentaje_confianza = confianza * 100

            return predicted_digit, porcentaje_confianza

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(
                f"Error: A ocurrido un error inesperado durante la predicción: {e}")
            return ""
