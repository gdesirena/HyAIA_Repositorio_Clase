import cv2
import numpy as np
import joblib
import os

# --- PREDICCION DE DIGITOS USANDO EL MODELO ENTRENADO ---
MODEL_FILE = 'Proyecto_Final/Proyecto/reconocimiento_digitos_svm.pkl'
IMG_SIZE = 28 # Usaremos este tamaño para que coincida con el entrenamiento

# Función para procesar la imagen a predecir usando los mismos criterios que en el modelo
def Procesar_Imagen(img_path):
    # Leemos la imagen en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Error: No se pudo cargar la imagen desde la ruta especificada: {img_path}")
        
    # Normalizamos el tamaño a IMG_SIZE x IMG_SIZE (28 X 28)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Convertimos a blanco y negro (Invertido)
    _, img_binary = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Aplanamos la imagen
    features = img_binary.flatten() / 255.0
    
    # Aseguramos que los datos sean en 2D para la predicción (1 fila, 784 columnas)
    return features.reshape(1, -1)

# CArgamos el modelo y predecimos
def Predecir_Digito():
    print("\ln--- Reconocedor de Dígitos ---")
    
    if not os.path.exists(MODEL_FILE):
        print(f"Error: No se encontró el modelo entrenado '{MODEL_FILE}'.")
        print("Asegúrate de ejecutar primero el archivo 'EntrenarModelo.py'.")
        return

    # 1. Cargar el modelo entrenado
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        print(f"Error: No fue posible cargar el modelo: {e}")
        return

    while True:
        # Ingresar la ruta de la imagen a predecir
        image_path = input("\nIngresa la ruta completa de la imagen ('salir' para terminar): ")
        
        if image_path.lower() == 'salir':
            break
            
        try:
            # Procesar la imagen
            features = Procesar_Imagen(image_path)
            
            # Realizar la predicción
            prediction = model.predict(features)
            
            # Mostrar el resultado
            predicted_digit = prediction[0]
            print("\n" + "="*36)
            print(f"El modelo predice que el dígito es: {predicted_digit}")
            print("="*36)

            # Esperamos a que el usuario presione una tecla.
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"Error: A ocurrido un error inesperado durante la predicción: {e}")
            cv2.destroyAllWindows()


if __name__ == "__main__":
    Predecir_Digito()