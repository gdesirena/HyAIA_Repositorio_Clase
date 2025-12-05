import pandas as pd
import numpy as np
import string
from collections import Counter
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt

class HyAIA:
    def __init__(self, df):
        self.data = df
        self.columns = df.columns
        self.data_binarios, self.binarios_columns = self.get_binarios()
        self.data_cuantitativos, self.cuantitativos_columns = self.get_cuantitativos()
        self.data_categoricos, self.categoricos_columns = self.get_categoricos()
        self.df_dqr = self.get_dqr()

    ##% Métodos para Análisis de Datos 
    #Método para obtener las columnas y dataframe binarios
    def get_binarios(self):
        col_bin = []
        for col in self.data.columns:
            if self.data[col].nunique() == 2:
                col_bin.append(col)        
        return self.data[col_bin], col_bin 

    #Método para obtener columnas y dataframe cuantitativos
    def get_cuantitativos(self):
        col_cuantitativas = self.data.select_dtypes(include='number').columns                
        return self.data[col_cuantitativas], col_cuantitativas

    #Método para obtener columnas y dataframe categóricos
    def get_categoricos(self):        
        col_categoricos = self.data.select_dtypes(exclude='number').columns
        col_cat = []
        for col in col_categoricos:
            if self.data[col].nunique()>2:
                col_cat.append(col)        
        return self.data[col_cat], col_cat
    
    # Limpieza de datos categóricos
    #def categoricos_limpieza(self):
    #    for col in self.categoricos_columns:
    #        self.data_categoricos[col] = self.data_categoricos[col].apply(remove_punctuation)

    def get_dqr(self):

        # Definimos las columnas categoricas y el máximo de categorias a mostrar
        cols_categoricas = set(self.categoricos_columns).union(set(self.binarios_columns))
        max_categorias = 10

        # DataFrames que se llenan inicialmente
        # --------------------------------------------

        #Nombre columna
        columns = pd.DataFrame(list(self.data.columns.values), columns=['Columns_Names'], index=list(self.data.columns.values))
        #Tipo de dato
        data_dtypes = pd.DataFrame(self.data.dtypes, columns=['Dtypes'])
        # Columnas con valor
        present_values = pd.DataFrame(self.data.count(), columns=['Present_Values'])
        # Columnas con nulos
        missing_values = pd.DataFrame(self.data.isnull().sum(), columns=['Missing_Values'])

        # DataFrames que se llenan durante el ciclo
        # --------------------------------------------
        
        unique_values = pd.DataFrame(columns=['Unique_Values']) # Valores únicos
        max_values = pd.DataFrame(columns=['Max_Values']) # Valor máximo
        min_values = pd.DataFrame(columns=['Min_Values']) # Valor mínimo
        desviacion_estandar = pd.DataFrame(columns=['Dev_Estandar']) # Desviación estándar
        media = pd.DataFrame(columns=['Media']) # Madia aritmetica

        is_categorical = pd.DataFrame(columns=['Is_categorical']) # Indica si es columna categórica
        categories_list = pd.DataFrame(columns=['Categorical_List']) # Muestra la lista de categorias si son 10 o menos

        # Ciclo para completar información por columna
        for col in self.data.columns:
            
            col_data = self.data[col]
            unique_count = col_data.nunique() # Se obtienen valores unicos 
            is_cat = col in cols_categoricas # True si la columna es categórica o binaria

            # Se obtiene la columna valores únicos
            unique_values.loc[col] = unique_count 

            # Se especifica si la columna es categorica
            is_categorical.loc[col] = [is_cat]

            # Se obtiene el listado de las categorías
            if is_cat:
                if unique_count <= max_categorias:
                    # Si tiene 10 o menos, listamos los valores
                    categories_list.loc[col] = [str(col_data.unique().tolist())] 
                else:
                    # Si tiene más de 10, mostramos el total
                    categories_list.loc[col] = [f'Más de {max_categorias} categorías. Total ({unique_count})']
            else:
                categories_list.loc[col] = ['N/A'] # Si no es categorica asignamos N/A

            # Calculamos max/min/mean/std si la columna es numérica
            try:
                if not is_cat: 
                    # Es numérica, aplicamos estadística
                    max_values.loc[col] = col_data.max()
                    min_values.loc[col] = col_data.min()
                    desviacion_estandar.loc[col] = col_data.std()
                    media.loc[col] = col_data.mean()
                else: 
                    # Es categórica ó binaria, asignamos N/A
                    max_values.loc[col] = 'N/A'
                    min_values.loc[col] = 'N/A'
                    desviacion_estandar.loc[col] = 'N/A'
                    media.loc[col] = 'N/A'

            except Exception:
                # Manejoamos control de errores por algún tema inesperado por el tipo de datos
                max_values.loc[col] = 'N/A'
                min_values.loc[col] = 'N/A'
                desviacion_estandar.loc[col] = 'N/A'
                media.loc[col] = 'N/A'
                pass

        # Unimos todos los DataFrames y retornamos todas las columnas
        self.df_dqr = (columns
            .join(data_dtypes)
            .join(present_values)
            .join(missing_values)
            .join(unique_values)
            .join(max_values)
            .join(min_values)
            .join(desviacion_estandar)
            .join(media)
            .join(is_categorical)
            .join(categories_list)            
            )
        
        return self.df_dqr
    
    # Método para obtener la media recortada
    def media_recortada(self, col, porcentaje_recorte): #agregar a la librería
        media_recortada = None
        n = len(self.data[col])
        r = int(n * porcentaje_recorte)
        datos_ordenados = sorted(self.data[col])
        datos_recortados = datos_ordenados[r:n-r] if n - 2*r > 0 else []
        if datos_recortados:
            media_recortada = sum(datos_recortados) / len(datos_recortados)
            print(f'Media recortada ({porcentaje_recorte * 100}%): {media_recortada}', )
        else:
            print('No hay suficientes datos para calcular la media recortada con este porcentaje.')
        return media_recortada

    #Creación de la función para obtener los outliers por el metodo IQR
    def MetodoIQR (self, n, features):
        outlier_list = []

        for column in features:
            #1st quartile (25%)
            Q1 = np.percentile(self.data[column], 25)
            #3st quartile (75%)
            Q3 = np.percentile(self.data[column], 75)

            # Calcular IQR
            IQR = Q3 - Q1

            # Definir rango de limite
            outlier_limit = 1.5 * IQR
            
            # Limite Inferior de iqr
            Li = Q1 - outlier_limit
            # Limite Superior de iqr
            Ls = Q3 + outlier_limit

            # Determinar la lista de outliers
            outlier_list_column = self.data[(self.data[column] < Li) | (self.data[column] > Ls)].index
            # Agregar a la lista de outliers
            outlier_list.extend(outlier_list_column)

        # Seleccionar las observaciones que contienen mas de cierto número de outliers
        outlier_list = Counter(outlier_list)
        print(outlier_list)
        multiple_outliers = list(k for k,v in outlier_list.items() if v >= n)
        
        return multiple_outliers

    #Método para encontrar la Desviación Estándar
    def Metodo_StDev(self, n, features):
        outlier_indices = []
        for column in features:
            #Calculamos la media y desviación estándar
            data_mean = self.data[column].mean()
            data_std = self.data[column].std()

            # Calculando el corte de la desviación estándar
            cut_off = 3 * data_std

            # Determinamos los indices de los outliers
            outlier_list_column = self.data[(self.data[column] < data_mean - cut_off) | (self.data[column] > data_mean + cut_off)].index

            # Agregamos los indices de la outliers obtenidos
            outlier_indices.extend(outlier_list_column)

        # Seleccionamos las observaciones que contienen más de ciertos nums de outliers
        outlier_ind = Counter(outlier_indices)
        multiple_outliers = list(k for k,v in outlier_ind.items() if v >= n)
        
        return multiple_outliers

    # Método de puntuación Z (Z_Scores)
    def Metodo_Z_score (self, n, features):
        outlier_list = []
        for column in features:
            #Calculamos la media y desviación estándar
            data_mean = self.data[column].mean()
            data_std = self.data[column].std()
            threshold = 3
            
            # Calculando el corte de la desviación estándar
            z_score = abs((self.data[column] - data_mean) / data_std)
            
            # Determinamos los indices de los outliers
            outlier_list_column = self.data[z_score > threshold].index

            # Agregamos los indices de la outliers obtenidos
            outlier_list.extend(outlier_list_column)

        # Seleccionamos las observaciones que contienen más de ciertos nums de outliers
        outlier_list = Counter(outlier_list)
        multiple_outliers = list(k for k,v in outlier_list.items() if v >= n)
        
        return multiple_outliers

    # Método de puntuación Z Modificado --Se debe agregar la libreria de Scipy: from scipy.stats import median_abs_deviation
    def Metodo_Z_ScoreMod (self, n, features): #Agregar a nuestra librería
        outlier_list = []
        for column in features:
            #Calculamos la media y desviación estándar de todo el dataframe
            data_median = self.data[column].median()
            data_std = self.data[column].std()
            threshold = 3
            
            # Calculando el mad z_score
            MAD = median_abs_deviation
            z_score = abs(0.6745 * (self.data[column] - data_median) / MAD(self.data[column]))
            
            # Determinamos los indices de los outliers
            outlier_list_column = self.data[z_score > threshold].index

            # Agregamos los indices de la outliers obtenidos
            outlier_list.extend(outlier_list_column)

        # Seleccionamos las observaciones que contienen más de ciertos nums de outliers
        outlier_list = Counter(outlier_list)
        multiple_outliers = list(k for k,v in outlier_list.items() if v >= n)
        
        return multiple_outliers

# Clase para realizar limpieza de datos    
class DataClean:
    def __init__(self):
        self.data = None
        #self.columns = df.columns
        #self.data_binarios, self.binarios_columns = self.get_binarios()
        #self.data_cuantitativos, self.cuantitativos_columns = self.get_cuantitativos()
        #self.data_categoricos, self.categoricos_columns = self.get_categoricos()
        #self.df_dqr = self.get_dqr()

    # remover signos de puntuación (Método estatico)
    @staticmethod 
    def remove_punctuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x
    
    # remover digitos
    @staticmethod
    def remove_digits(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.digits)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x
    
    # remover espacios en blanco
    @staticmethod
    def remove_whitespace(x):
        try:
            x = ' '.join(x.split())
        except:
            pass
        return x

    # convertir a minisculas
    @staticmethod
    def lower_text(x):
        try:
            x = x.lower()
        except:
            pass
        return x

    #convertir a mayusculas
    @staticmethod
    def upper_text(x):
        try:
            x = x.upper()
        except:
            pass
        return x

    # Función que convierta a mayúsculas la primera letra
    @staticmethod
    def capitalize_text(x):
        try:
            x = x.capitalize()
        except:
            pass
        return x
        
    # reemplazar texto
    @staticmethod
    def replace_text(x, to_replace, replacement):
        try:
            x = x.replace(to_replace, replacement)
        except:
            pass
        return x

# Clase Máquinas de Vector Soporte (Support Vector Machine)
class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr  #Tasa de aprendizaje utilizada para actualizar los parámetros durante la optimización.
        self.lambda_param = lambda_param  #Parámetro de regularización que controla la penalización sobre los pesos para evitar sobreajuste.
        self.n_iters = n_iters  #Número de iteraciones (épocas) para el proceso de entrenamiento.
        self.w = None  #Vector de pesos del hiperplano que separa las clases.
        self.b = None  #Término de sesgo (bias) que ajusta la posición del hiperplano.

    def fit(self, X, y):
        X = np.array(X)
        y = np.where(np.array(y) == 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condicion = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condicion:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        X = np.array(X)
        approx = np.dot(X, self.w) + self.b
        return np.where(approx >= 0, 1, 0)
    
#Clase para Análisis de Componentes Principales
class PCA:
    def __init__(self, n_components):
        self.n_components=n_components
        self.components = None
        self.mean = None
        self.ratio = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        #1.- Calcular la covarianza de los datos X
        cov = np.cov(X.T)
        
        #2. Calcular los eigenvalores y eigenventores
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        #3. Ordenar los eigenvalores con los eigenvectores asociados  
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:,self.n_components]
        self.ratio = eigenvalues / np.sum(eigenvalues)
                
    def transform(self,X):
        #Proyección al nuevo espacio
        transformacion = np.dot(X, self.components.T)
        return transformacion
    
    def get_ratio(self):        
        return self.ratio[:self.n_components]

    def plot_explained_variance_ratio(self, percentage=1):
        #Plotear la varianza explicada acumulada
        plt.figure(figsize=(8,6))
        plt.plot(np.arange(1, len(self.ratio) + 1), np.comsum(self.ratio))
        plt.hlines(y = percentage, xmin = 1, xmax = len(self.ratio), color = 'r', linestyle = 'dashed')
        plt.xlabel('Numero de componentes')
        plt.ylabel('Varianza acumulada')        
        plt.show()