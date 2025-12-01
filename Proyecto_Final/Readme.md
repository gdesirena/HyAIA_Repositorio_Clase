# Proyectos Finales de Aplicación

Este directorio contiene los proyectos finales desarrollados para la asignatura Herramientas y Aplicaciones de la IA. Las instrucciones de cada proyecto están en un cuaderno Jupyter (.ipynb) y contiene la descripción del problema. A continuación se incluye una descripción de cada proyecto, instrucciones para ejecutar los notebooks, dependencias recomendadas y buenas prácticas.

## Resumen de proyectos

1. Clasificación Automática de Objetos
	- Notebook: [Proyecto Final (clasificación automática de objetos).ipynb](https://github.com/gdesirena/HyAIA_Repositorio_Clase/blob/main/Proyecto_Final/Proyecto%20Final%20(clasificaci%C3%B3n%20autom%C3%A1tica%20de%20objetos).ipynb)
	- Propósito: Construir y evaluar un clasificador de imágenes (por ejemplo CNN) capaz de reconocer diferentes categorías de objetos.
	- Entradas: Conjunto de imágenes etiquetadas (puede usarse CIFAR, una colección personalizada, o subcarpetas organizadas por clase).
	- Salidas: Modelo entrenado, matriz de confusión, métricas (accuracy, precision, recall), ejemplos de predicción.
	- Notas de ejecución: revisar las celdas de carga de datos para apuntar a la ruta correcta del dataset (por ejemplo `data/images/`).

2. Clasificación de Géneros de Música
	- Notebook: [Proyecto Final (Clasificación de Generos de Música).ipynb](https://github.com/gdesirena/HyAIA_Repositorio_Clase/blob/main/Proyecto_Final/Proyecto%20Final%20(Clasificaci%C3%B3n%20de%20Generos%20de%20M%C3%BAsica).ipynb)
	- Propósito: Extraer features de audio (MFCCs, espectrogramas) y entrenar un clasificador que identifique el género musical.
	- Entradas: Archivos de audio (WAV/MP3). Dataset típico: GTZAN o datasets propios organizados por carpeta de género.
	- Salidas: Modelo de clasificación, curva ROC, matriz de confusión, ejemplos de audio con predicción.
	- Dependencias específicas: `librosa` para extracción de características de audio.

3. Detección de Género y Edad
	- Notebook: [Proyecto Final (Detección de género y edad).ipynb](https://github.com/gdesirena/HyAIA_Repositorio_Clase/blob/main/Proyecto_Final/Proyecto%20Final%20(Detecci%C3%B3n%20de%20g%C3%A9nero%20y%20edad).ipynb)
	- Propósito: Detectar y estimar el género y la edad de personas a partir de imágenes faciales.
	- Entradas: Imágenes faciales (datasets posibles: IMDB-WIKI, Adience, o dataset propio). Requiere detección de rostro previa (Haar cascades o MTCNN) y normalización.
	- Salidas: Modelos para clasificación de género y estimación de edad (regresión o clasificación por rangos), métricas de desempeño.
	- Dependencias específicas: `opencv-python`, `tensorflow`/`keras` o `pytorch` según implementación.

4. Reconocimiento de Caras usando PCA
	- Notebook: [Proyecto Final (Reconocimiento de Caras Usando PCA).ipynb](https://github.com/gdesirena/HyAIA_Repositorio_Clase/blob/main/Proyecto_Final/Proyecto%20Final%20(Reconocimiento%20de%20Caras%20Usando%20PCA).ipynb)
	- Propósito: Implementar reconocimiento facial clásico usando reducción de dimensionalidad (PCA / Eigenfaces) y un clasificador simple (k-NN, SVM).
	- Entradas: Imágenes de caras etiquetadas; recomendable tener un dataset organizado por persona.
	- Salidas: Representación mediante autovectores (eigenfaces), modelos y métricas de reconocimiento.
	- Notas: útil para entender métodos tradicionales antes de pasar a redes profundas.

5. Reconocimiento de Dígitos a Mano
	- Notebook: [Proyecto Final (Reconocimiento de Digitos a Mano).ipynb](https://github.com/gdesirena/HyAIA_Repositorio_Clase/blob/main/Proyecto_Final/Proyecto%20Final%20(Reconocimiento%20de%20Digitos%20a%20Mano).ipynb)
	- Propósito: Resolver el problema de reconocimiento de dígitos manuscritos (por ejemplo usando MNIST) con enfoques basados en ML clásico o redes neuronales.
	- Entradas: Imágenes de dígitos (MNIST o dataset propio).
	- Salidas: Clasificador entrenado, métricas de evaluación y ejemplos de predicción.

## Instrucciones generales 

1. Entorno recomendado
	- Python 3.8+ (se recomienda 3.8–3.11).
	- Crear un entorno virtual (venv/conda) y activar antes de instalar dependencias.

	PowerShell (Windows):

	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	pip install --upgrade pip
	```

2. Instalar dependencias
	- Si existe un archivo `requirements.txt` en la raíz del proyecto, instálelo con:

	```powershell
	pip install -r requirements.txt
	```

	- Si no hay `requirements.txt`, instale las bibliotecas mínimas comúnmente usadas:

	```powershell
	pip install jupyterlab notebook numpy pandas matplotlib seaborn scikit-learn opencv-python pillow
	pip install tensorflow   # o torch, según el notebook
	pip install librosa     # sólo si trabaja con audio
	```

3. Abrir y ejecutar notebooks
	- Inicie Jupyter Lab/Notebook desde la raíz del repositorio (o desde la carpeta `Proyecto_Final`):

	```powershell
	jupyter lab
	```

	- Abra el cuaderno deseado y ejecute las celdas en orden. Preste atención a las celdas iniciales donde se configuran las rutas a datasets y parámetros.

4. Estructura esperada de datos
	- Coloque los datasets en una carpeta `data/` dentro de `Proyecto_Final` o modifique las rutas en las celdas del notebook.
	- Para datasets grandes, añada sólo los scripts/notas para descargar los datos (no subir datasets pesados al repositorio).

## Notas sobre datasets y rutas

- Si un notebook requiere descargar datos, normalmente incluye la instrucción o enlace para obtenerlos. Revise la primera sección del notebook.
- Para reproducibilidad, mantenga la estructura:

  Proyecto_Final/
  ├─ data/
  │  ├─ objetos/
  │  ├─ music/
  │  ├─ faces/
  │  └─ mnist/
  └─ Proyecto Final (...).ipynb

## Buenas prácticas y recomendaciones

- Lea las celdas de configuración al inicio de cada notebook y adapte las rutas a su entorno.
- Use entornos virtuales para evitar conflictos de dependencias.
- Para entrenamientos largos, considere usar una máquina con GPU o reducir el tamaño del dataset para pruebas rápidas.
- Añada un `requirements.txt` si va a compartir o desplegar el proyecto para facilitar la instalación.

## Siguientes pasos sugeridos (proactivo)

- Crear un `requirements.txt` mínimo para cada proyecto o uno global en la raíz.
- Añadir instrucciones específicas para cada notebook sobre dónde descargar datasets grandes.
- Incluir scripts de preprocesado que automaticen la preparación de datos (`scripts/download_data.py`, `scripts/preprocess.py`).

