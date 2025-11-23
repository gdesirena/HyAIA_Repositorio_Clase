import pandas as pd
import numpy as np
import string

#Clase que representa todo lo que veremos
class HyAIA:
    def __init__(self, df): #Método constructor
        self.data = df #Atributos (Df original)
        self.columns = df.columns #Columnas originales
        self.data_binarios, self.binarios_columns = self.get_binarios() #Atributos para binarios
        self.data_cuantitativos, self.cuantitativos_columns = self.get_cuantitativos() #Atributos para cuantitativos
        self.data_categoricos, self.categoricos_columns = self.get_categoricos() #Atributos categoricos
        self.df_dqr = self.get_dqr() #
        
    ##% Métodos para Análisis de Datos 
    #Método para obtener las columnas y dataframe binarios
    def get_binarios(self):
        col_bin = [] #Lista vacía
        for col in self.data.columns:
            if self.data[col].nunique() == 2: #Numeros de elementos de valor único (" binaria)
                col_bin.append(col)
        return self.data[col_bin],col_bin #Retornar una tupla

    
    #Método para obtener columnas y dataframe cuantitativos
    def get_cuantitativos(self):
        col_cuantitativas = self.data.select_dtypes(include='number').columns #Selecciono los tipos de datos que tienen numeros    
        return self.data[col_cuantitativas], col_cuantitativas
        
       
    #Método para obtener columnas y dataframe categóricos
    def get_categoricos(self):
        col_categoricos = self.data.select_dtypes(exclude ='number').columns #Excluye los numeros y devuelve columnas
        col_cat = [] #Lista vacía
        for col in col_categoricos:
            if self.data[col].nunique()>2: #No valores numericos =< a 2
                col_cat.append(col) #Columnas categoricas s/binarias
        return self.data[col_cat], col_cat

    #Método para remover signos de puntuación en todas las columnas
   # def categoricos_limpieza(self):
    #    for col in self.categoricos_columns:
     #       self.data_categoricos[col] = self.data_categoricos[col].apply(remove_punctuation) #Autoasignacion

   
    #Metodos de limpieza de datos.
    #Remover signos de puntuación
    @staticmethod #Metodo estatico no requiere el self
    def remove_punctuation(x):
        try: #Capturar errores
        #Elemento de entrada, unir todos los caracteres si el caracter no es signo de puntuacion
            x = ''.join(ch for ch in x if ch not in string.punctuation) 
        except:
            print(f'{x} no es una cadena de caracteres')
            pass #Dejar pasar la función, no hara nada
        return x

    #Remover dígitos
    @staticmethod 
    def remove_digits(x):
        try:
            x=''.join(ch for ch in x if ch not in string.digits)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x

    # remover espacios en blanco
    @staticmethod 
    def remove_whitespace(x):
        try:
            x=' '.join(x.split()) #split generando las listas de cadenas de caracteres y despues las uno
        except:
            pass
        return x

    #Convertir a minisculas
    @staticmethod
    def lower_text(x):
        try:
            x = x.lower()
        except:
            pass
        return x

    #Convertir a mayusculas
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
    
    # Reemplazar texto
    @staticmethod
    def replace_text(x,to_replace, replacement):
        try:
            x = x.replace(to_replace, replacement)
        except:
            pass
        return x


    ## MÉTODO PARA UN REPORTE DE CALIDAD DE DATOS 
#Método de la clase

    def get_dqr(self):
    
            #% Lista de variables de la base de datos
        columns = pd.DataFrame(list(self.data.columns.values), columns=['Columns_Names'], 
                                   index=list(self.data.columns.values))  #DataFrame con los nombres de las columnas y se usan como índice.
        
            #Lista de tipos de datos del dataframe
        data_dtypes = pd.DataFrame(self.data.dtypes, columns=['Dtypes']) #Obtiene el tipo de dato de cada columna 
            
            #Lista de valores presentes
        present_values = pd.DataFrame(self.data.count(), columns=['Present_values']) #Count conteo de los registos con valor 
            
            #Lista de valores missing (Valores faltantes/nulos nan)
        missing_values = pd.DataFrame(self.data.isnull().sum(), columns=['Missing_values']) #Cuenta los valores faltantes.
            
            #Valores unicos de las columnas
        unique_values = pd.DataFrame(columns=['Unique_values'])
        for col in list(self.data.columns.values):
            unique_values.loc[col] = [self.data[col].nunique()] #Cuenta cuántos valores distintos tiene cada columna
            
            ## INFORMACIÓN ESTADÍSTICA
    
        #Columna booleana: (Categórica)
        is_categorical = pd.DataFrame(columns=['Is_categorical'])
        for col in list(self.data.columns.values):
            if self.data[col].dtype == 'object':
                is_categorical.loc[col] = [True] #True si es de tipo texto (object)
            else:
                is_categorical.loc[col] = [False] #False si es numérica
                    
            # Lista de valores máximos
        max_values = pd.DataFrame(columns=['Max_values'])
        for col in list(self.data.columns.values):
            if is_categorical.loc[col, 'Is_categorical'] == True:
                max_values.loc[col] = ['False'] #Si es categórica pone "False"
            else:
                max_values.loc[col] = [self.data[col].max()] #Si es numérica calcula el máximo
    
    
    
            #for col in list(data.columns.values):
               # try:
                   # max_values.loc[col] = [data[col].max()]
               # except:
                  #  max_values.loc[col] = ['N/A']
               # pass
            
            # Lista de valores mínimos
        min_values = pd.DataFrame(columns=['Min_values'])
        for col in list(self.data.columns.values):
            if is_categorical.loc[col, 'Is_categorical'] == True: 
                 min_values.loc[col] = ['False'] #Si es categórica pone "False"
            else:
                 min_values.loc[col] = [self.data[col].min()] #Si es numérica calcula el mínimo
    
             
           # for col in list(data.columns.values):
            #    try:
             #       min_values.loc[col] = [data[col].min()]
              #  except:
               #     min_values.loc[col] = ['N/A']
                #pass
            
            # Lista de valores promedio (media)
        mean_values = pd.DataFrame(columns=['Mean_values'])
        for col in list(self.data.columns.values):
            if is_categorical.loc[col, 'Is_categorical'] == True:
                mean_values.loc[col] = ['False'] #Si es categórica "False"
            else:
                mean_values.loc[col] = [self.data[col].mean()] #Si es numérica calcula la media
    
    
            #for col in list(data.columns.values):
             #   try:
              #      mean_values.loc[col] = [data[col].mean()]
               # except:
                #    mean_values.loc[col] = ['N/A']
                #pass
            
            # Lista de desviación estándar
        std_values = pd.DataFrame(columns=['Std_values'])
        for col in list(self.data.columns.values):
            if is_categorical.loc[col, 'Is_categorical'] == True:
                std_values.loc[col] = ['False']
            else:
                std_values.loc[col] = [self.data[col].std()] #Solo para columnas numéricas calcula la desviación estándar
    
    
            #for col in list(data.columns.values):
             #   try:
              #      std_values.loc[col] = [data[col].std()]
               # except:
                #    std_values.loc[col] = ['N/A']
                #pass
            
            # Percentil 25
        percentile_25 = pd.DataFrame(columns=['25%'])
        for col in list(self.data.columns.values):
            try:
                percentile_25.loc[col] = [self.data[col].quantile(0.25)]
            except:
                percentile_25.loc[col] = ['N/A']
            pass
            
            # Percentil 50 (mediana)
        percentile_50 = pd.DataFrame(columns=['50%'])
        for col in list(self.data.columns.values):
            try:
                percentile_50.loc[col] = [self.data[col].quantile(0.50)]
            except:
                percentile_50.loc[col] = ['N/A']
            pass
            
            # Percentil 75
        percentile_75 = pd.DataFrame(columns=['75%'])
        for col in list(self.data.columns.values):
            try:
                percentile_75.loc[col] = [self.data[col].quantile(0.75)]
            except:
                percentile_75.loc[col] = ['N/A']
            pass
                    
                 
    
            # Columna: Categorías (solo si es categórica y <= 10 categorías)
        categories = pd.DataFrame(columns=['Categories'])
        for col in list(self.data.columns.values):
            if self.data[col].dtype == 'object':
                uniques = self.data[col].dropna().unique()
                if len(uniques) <= 10:
                    categories.loc[col] = [', '.join(map(str, uniques))]
                else:
                    categories.loc[col] = ['N/A']
            else:
                categories.loc[col] = ['N/A']
    
        
    
            # Salida final
        return (
            columns
            .join(data_dtypes)
            .join(present_values)
            .join(missing_values)
            .join(unique_values)
            .join(is_categorical)
            .join(max_values)
            .join(min_values)
            .join(mean_values)
            .join(percentile_25)
            .join(percentile_50)
            .join(percentile_75)
            .join(std_values)
            .join(categories)
        )
        
    
       

    