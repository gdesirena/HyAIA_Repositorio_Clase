# Mi Primer Librería de Análisis de Datos en Python
# Autor: Gaddiel interprete Gabriela Ramírez
# Fecha: 18 de noviembre de 2025
import pandas as pd
import numpy as np
import string


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
            if self.data[col].nunique() > 2:
                col_cat.append(col)
        return self.data[col_cat], col_cat

    #REPORTE DE CALIDAD DE DATOS (DQR)
    def get_dqr(self):
        #% Lista de variables de la base de datos
        columns = pd.DataFrame(list(self.data.columns.values), columns=['Columns_Names'],
                               index=list(self.data.columns.values))
    
        #Lista de tipos de datos del dataframe
        data_dtypes = pd.DataFrame(self.data.dtypes, columns=['Dtypes'])
    
        #Lista de valores presentes
        present_values = pd.DataFrame(self.data.count(), columns=['Present_values'])
    
        #Lista de valores missing (Valores faltantes/nulos nan)
        missing_values = pd.DataFrame(self.data.isnull().sum(), columns=['Missing_values'])
    
        #Valores unicos de las columnas
        unique_values = pd.DataFrame(columns=['Unique_values'])
        for col in list(self.data.columns.values):
            unique_values.loc[col] = [self.data[col].nunique()]
    
       # ====================================================TAREA 1 =====================================
        #Lista de valores Boleanos
        #1. Columna Boleana ls_categorical
        ls_categorical = pd.DataFrame(columns=['ls_categorical'])
        for col in list(self.data.columns.values):
          try:
              ls_categorical.loc[col] = [self.data[col].dtype =='object' or self.data[col].dtype.name == 'category']
          except:
              ls_categorical.loc[col] = [False]
              
        #********************************************************************
        #Valores Únicos < 10
        #2. Columna de Categorias de Valores Únicos < 10 
        
        categorias_values = pd.DataFrame(columns=['Categoricas'])
        for col in list(self.data.columns.values):
            es_categorical = ls_categorical.loc[col][0] == True
            if es_categorical:
                uniques = self.data[col].dropna().unique()
                if len(uniques) <= 10:
                    categorias_values.loc[col] = [list(uniques)]
                else:
                    categorias_values.loc[col] = [None]
            else:
                categorias_values.loc[col] = [None]

       # INFORMACIÓN ESTADÍSTICA
       #********************************************************************
       #3. Análisis estadístico (max, min, mean, std) para columnas númericas Is_categorical == False
        
        #Lista de valores máximos
        max_values = pd.DataFrame(columns=['Max_values'])
        
        for col in self.data.columns.values:
            if ls_categorical.loc[col][0] == False:
                try:
                    max_values.loc[col] = [self.data[col].max()]
                except:
                    max_values.loc[col] = ['NA']
            else:
                max_values.loc[col] = ['NA']
    
        #Lista de valores mínimos
        min_values = pd.DataFrame(columns=['Min_values'])
        
        for col in self.data.columns:
            if ls_categorical.loc[col][0] == False:
                try:
                    min_values.loc[col] = [self.data[col].min()]
                except:
                    min_values.loc[col] = ['NA']
            else:
                min_values.loc[col] = ['NA'] 
       
        #Lista de Valores con media
        mean_values = pd.DataFrame(columns=['Mean_values'])
        
        for col in self.data.columns.values:
            if ls_categorical.loc[col][0] == False: 
                try:
                    mean_values.loc[col] = [self.data[col].mean()]
                except:
                    mean_values.loc[col] = ['NA']
            else:
                mean_values.loc[col] = ['NA'] 
        
        #Lista de valores con desviación estandar
        dstd_values = pd.DataFrame(columns=['Dstd_values'])
        
        for col in self.data.columns:
            if ls_categorical.loc[col][0] == False:
              try:
                  dstd_values.loc[col] = [self.data[col].std()]
              except:
                  dstd_values.loc[col] = ['NA']
            else:
                dstd_values.loc[col] = ['NA']
    #=======================================================FIN TAREA=========================================================    
   
        
        
    # =========AGREGAR COLUMNAS=====================
        return (columns
                .join(data_dtypes)
                .join(present_values)
                .join(missing_values)
                .join(unique_values)
                .join(ls_categorical)
                .join(categorias_values)
                .join(max_values)
                .join(min_values)
                .join(mean_values)
                .join(dstd_values)
               )
    
    @staticmethod
    # remover signos de puntuación
    def remove_punctuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x
    
    @staticmethod
    def remove_digits(x):
        try:
            x=''.join(ch for ch in x if ch not in string.digits)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x
    
    @staticmethod
    # remover espacios en blanco
    def remove_whitespace(x):
        try:
            x=' '.join(x.split())
        except:
            pass
        return x
    
    @staticmethod
    # convertir a minisculas
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
    def replace_text(x,to_replace, replacement):
        try:
            x = x.replace(to_replace, replacement)
        except:
            pass
        return x

'''
    #def categoricos_limpieza(self):
    #    for col in self.cateforicos_columns:
            #self.data_categoricos[col] = self.data_categoricos[col].apply(remove_punctuation)
'''