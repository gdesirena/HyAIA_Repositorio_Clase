import cv2
import numpy as np
import os
from tqdm import tqdm # Lo uso para mostrar avance en la carga de imagenes

# --- CARGA DE IMAGENES DESDE LOS FOLDERS ---
# Folder donde se encuentran las imagenes de (0 a 9)

DATA_DIR = 'saved_images'

# Tamaño para redimensionar las imágenes y sean equitativas
IMG_SIZE = 28

# Ruta donde guardamos el dataset procesado de NumPy
OUTPUT_FILE = 'imagenesnumeros_data.npz'

#Función para realizar la carga de las imagenes y crear el arreglo con features (X) y etiquetas (y)
def load_and_process_images():
    data = [] # Se almacenarán las características (Todos los píxeles de cada dígito)
    labels = [] # Se almacenarán las etiquetas (dígitos)

    # Obtenemos los nombres de las subcarpetas (los dígitos 0-9) y también lasusaremos como etiquetas ó targets del modelo
    digit_folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    print(f"\nCarpetas de dígitos encontradas: {digit_folders}")

    # Para cada uno de los folders procesamos las imagenes
    for digit in digit_folders:
        folder_path = os.path.join(DATA_DIR, digit)
        label = int(digit) # La etiqueta es el nombre de la carpeta
        
        # Iteramos por cada imágen en la carpeta del dígito actual
        for filename in tqdm(os.listdir(folder_path), desc=f"Procesando dígito {digit}"):
            # Los tipos de imagenes soportados son: .png, .jpg y .jpeg.   Entre más cantidad de imagenes por cada dígito mejor accuracy lograremos tener.
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                
                try:
                    # Leemos la imagen en escala de grises para estandarizar el color
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        print(f"Advertencia: No se pudo cargar la imagen {img_path}")
                        continue
                        
                    # Normalizamos el tamaño a IMG_SIZE x IMG_SIZE (28 X 28)
                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

                    # Convertimos a Blanco y Negro para que el módelo funcione mejor con el fondo negro (0) y el dígito blanco (255)
                    _, img_binary = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    
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
    np.savez_compressed(OUTPUT_FILE, X=X, y=y)
    print(f"Archivo de datos guardado exitosamente en {OUTPUT_FILE}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: El folder '{DATA_DIR}' no existe.")
        print("Favor de crearla y añadir los subcarpetas (0 a 9) con las imágenes de dígitos.")
    else:
        print(f"Folder '{DATA_DIR}' encontrado. Y se cargaran las imagenes...")
        load_and_process_images()