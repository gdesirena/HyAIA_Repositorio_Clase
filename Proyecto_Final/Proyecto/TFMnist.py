
from pathlib import Path
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info messages


class TFMnist:
    def __init__(self):
        self.Convolutional_Model = None
        self.history_data = None
        self.history_path = None
        self.test_data_path = None
        self.X_test = None
        self.y_test = None
        self.modelo_path = 'modelo_mnist.h5'
        self.cargar_modelo_si_existe()
        self.IMG_SIZE = 28      
        
    def normalize_images(self, images):
        images = images.astype('float32')
        images /= 255

        return images

    def loadImagesTF(self):

        if self.Convolutional_Model != None:
            return
        
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = self.normalize_images(X_train)
        X_test = self.normalize_images(X_test)
        # Redimensionar la matrix X_train de las imagenes a un tensor de (m,28,28,1)
        X_train = X_train.reshape(
            X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        # Redimensionar la matrix X_test de las imagenes a un tensor de (m,28,28,1)
        X_test = X_test.reshape(
            X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        # Creando el modelo
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                  activation='relu', input_shape=X_train.shape[1:]))
        model.add(tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        # Capas FC densas
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        # Capa de salida
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        print(model.summary())

        model.compile(loss='categorical_crossentropy',
                      optimizer='Adadelta', metrics=['accuracy'])

        # CAPTURA EL HISTORIAL DE ENTRENAMIENTO
        history = model.fit(X_train, y_train,
                  batch_size=32,
                  epochs=15,
                  verbose=1,
                  validation_data=(X_test, y_test))
        score = model.evaluate(X_test, y_test, verbose=1)
        self.Convolutional_Model = model
        self.guardar_modelo_h5()

        self.history_path = 'modelo_mnist_history.json'
        # Guardar historial del modelo como JSON
        try:
            with open(self.history_path, 'w') as f:
                json.dump(history.history, f)
            print(f"✅ Historial de entrenamiento guardado en: {self.history_path}")
        except Exception as e:
            print(f"❌ Error al guardar el historial: {e}")

        # Almacena el historial y los datos de prueba
        self.X_test = X_test
        self.y_test = y_test
        self.history = history

        # Guardar los datos de prueba para graficarlos desde pantalla
        self.test_data_path = 'mnist_test_data.npz'
        try:
            np.savez(self.test_data_path, X_test=X_test, y_test=y_test)
            print(f"✅ Datos de prueba guardados en: {self.test_data_path}")
        except Exception as e:
            print(f"❌ Error al guardar los datos de prueba: {e}")

    def guardar_modelo_h5(self):
        """Guarda el modelo completo en formato .h5"""
        if self.Convolutional_Model is None:
            print("❌ No hay modelo para guardar")
            return

        self.Convolutional_Model.save(self.modelo_path)
        print(f"✅ Modelo guardado en: {self.modelo_path}")
        print(f"   Incluye: arquitectura, pesos, configuración de entrenamiento")

    def cargar_modelo_si_existe(self):
        """
        Verifica si el archivo del modelo existe y lo carga
        """
        # Usar pathlib para manejo robusto de rutas
        modelo_file = Path(self.modelo_path)
        self.history_path = 'modelo_mnist_history.json'
        self.test_data_path = 'mnist_test_data.npz'

        if modelo_file.exists():
            print(f"✅ Encontrado archivo: {self.modelo_path}")
            print("⚡ Cargando modelo...")

            try:
                self.Convolutional_Model = tf.keras.models.load_model(
                    str(modelo_file))
                print("🎯 Modelo cargado exitosamente!")
                print(f"📊 Nombre del modelo: {self.Convolutional_Model.name}")

                # ⚡ Cargar el Historial ⚡
                history_file = Path(self.history_path)
                if history_file.exists():
                    with open(self.history_path, 'r') as f:
                        # Se carga como un diccionario y se asigna al atributo
                        self.history_data = json.load(f)
                    print(f"✅ Historial cargado desde: {self.history_path}")

                else:
                    self.history_data = None
                    print("⚠️ Archivo de historial no encontrado. No se podrán mostrar las gráficas de entrenamiento.")

                # ⚡ Cargar los datos de prueba ⚡
                test_data_file = Path(self.test_data_path)
                if test_data_file.exists():
                    try:
                        # Usamos np.load para cargar el archivo .npz
                        data = np.load(self.test_data_path)
                        self.X_test = data['X_test']
                        self.y_test = data['y_test']
                        print(f"✅ Datos de prueba cargados ({self.X_test.shape[0]} muestras).")
                    except Exception as e:
                        self.X_test = None
                        self.y_test = None
                        print(f"❌ Error al cargar los datos de prueba: {e}")
                else:
                    self.X_test = None
                    self.y_test = None
                    print("⚠️ Archivo de datos de prueba no encontrado.")

                # Opcional: mostrar arquitectura
                self.mostrar_info_modelo()

                return True

            except Exception as e:
                self.Convolutional_Model = None
                #self.history_data = None
                print(f"⚡ Error al cargar modelo {e}")
        else:
            self.Convolutional_Model = None
            #self.history_data = None
            print("⚡ Modelo no encontrado")

    def mostrar_info_modelo(self):
        """Muestra información básica del modelo cargado"""
        if self.Convolutional_Model:
            print("\n📋 Información del modelo cargado:")
            print(f"   Capas: {len(self.Convolutional_Model.layers)}")
            print(
                f"   Parámetros totales: {self.Convolutional_Model.count_params():,}")

            # Mostrar tipo de cada capa
            print("\n   Arquitectura:")
            for i, layer in enumerate(self.Convolutional_Model.layers):
                print(f"     {i+1}. {layer.name} ({layer.__class__.__name__})")

    def Procesar_Imagen(self, img_path):
        """
        Procesa una imagen para que sea compatible con el modelo MNIST

        Args:
            img_path: Ruta a la imagen

        Returns:
            numpy array de forma (1, 28, 28, 1)
        """
        # Leemos la imagen en escala de grises
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(
                f"Error: No se pudo cargar la imagen desde la ruta especificada: {img_path}")

        # Normalizamos el tamaño a IMG_SIZE x IMG_SIZE (28 X 28)
        img_resized = cv2.resize(
            img, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Invertir colores si es necesario
        # ¡IMPORTANTE! Usar img_resized, no img
        if np.mean(img_resized) > 127:
            img_resized = 255 - img_resized

        # Normalizar
        img_normalized = self.normalize_images(img_resized)

        # Verificar que tenga el tamaño correcto
        print(f"🔍 Forma después de normalizar: {img_normalized.shape}")

        if img_normalized.shape != (28, 28):
            print(
                f"⚠️  Advertencia: La imagen no tiene tamaño 28x28, tiene {img_normalized.shape}")
            print("   Forzando redimensionamiento...")
            img_normalized = cv2.resize(
                img_normalized, (28, 28), interpolation=cv2.INTER_AREA)

        # Añadir dimensiones
        img_procesada = img_normalized.reshape(1, 28, 28, 1)

        print(f"✅ Forma final: {img_procesada.shape}")
        print(
            f"✅ Rango de valores: [{img_procesada.min():.3f}, {img_procesada.max():.3f}]")

        return img_procesada

    def graficar_asertividad(self):
        """
        Grafica la precisión (Accuracy) y la pérdida (Loss) durante el entrenamiento.
        """
        if not hasattr(self, 'history_data') or self.history_data is None:
            print("❌ El modelo debe ser entrenado primero para tener el historial.")
            return

        hist = self.history_data

        # Figura para Precisión y Pérdida
        plt.figure(figsize=(14, 5))

        # Subplot 1: Precisión
        plt.subplot(1, 2, 1)
        plt.plot(hist['accuracy'], label='Precisión Entrenamiento')
        plt.plot(hist['val_accuracy'], label='Precisión Validación')
        plt.title('Precisión del Modelo')
        plt.ylabel('Precisión')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True)

        # Subplot 2: Pérdida
        plt.subplot(1, 2, 2)
        plt.plot(hist['loss'], label='Pérdida Entrenamiento')
        plt.plot(hist['val_loss'], label='Pérdida Validación')
        plt.title('Pérdida del Modelo')
        plt.ylabel('Pérdida')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.show()

    def graficar_confiabilidad(self):
        """
        Genera la Matriz de Confusión para visualizar la confiabilidad por clase.
        """
        if self.Convolutional_Model is None or self.X_test is None or self.y_test is None:
            print("❌ El modelo, X_test o y_test no están disponibles. Se requiere cargar el modelo y los datos.")
            return

        print("Calculando predicciones en datos de prueba...")
        # Obtener las probabilidades de predicción para las imágenes de prueba
        y_pred_probs = self.Convolutional_Model.predict(self.X_test, verbose=0)

        # Convertir probabilidades a etiquetas de clase (el índice con la mayor probabilidad)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        # Convertir las etiquetas de prueba one-hot a etiquetas de clase simples
        y_true_classes = np.argmax(self.y_test, axis=1)

        # Calcular la Matriz de Confusión
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Graficar la Matriz de Confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicción')
        plt.ylabel('Valor Verdadero')
        plt.title('Matriz de Confusión (Confiabilidad por Clase)')
        plt.show()

    def Predecir_Digito(self,  parImagePath):
        try:
            # Procesar la imagen
            features = self.Procesar_Imagen(parImagePath)

            # Realizar la predicción
            prediction = self.Convolutional_Model.predict(features)

            print(f"Prediccion: {prediction}")
            digito = np.argmax(prediction)

            prediction_vector = prediction[0]
            digito = np.argmax(prediction_vector)
            confianza = prediction_vector[digito]
            porcentaje_confianza = confianza * 100

            return digito, porcentaje_confianza

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(
                f"Error: A ocurrido un error inesperado durante la predicción: {e}")
            return ""

        # 1. Cargar el modelo entrenado
        try:
            # Procesar la imagen
            x_prueba = self.X_testT[0].reshape(28, 28)
            x_prueba.shape

            prediction = self.Convolutional_Model.predict(
                self.X_testT[0].reshape(1, 28, 28, 1))

            print(f"Prediccion: {prediction}")
            digito = np.argmax(prediction)
            print(f"Vector: {prediction}")
            print(f"Dígito correspondiente: {digito}")

            return digito

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(
                f"Error: A ocurrido un error inesperado durante la predicción: {e}")
            return ""
