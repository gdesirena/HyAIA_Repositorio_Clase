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
            if self.data[col].nunique()>2:
                col_cat.append(col)
        return self.data[col_cat], col_cat
        
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
        
        # TAREA 1: Columna booleana para identificar si es categórica
        is_categorical = pd.DataFrame(columns=['Is_categorical'])
        for col in list(self.data.columns.values):
            # Si el tipo de dato no es numérico, es categórica
            is_categorical.loc[col] = [not pd.api.types.is_numeric_dtype(self.data[col])]
        
        # TAREA 2: Columna con las categorías (si tiene <= 10 valores únicos)
        categories = pd.DataFrame(columns=['Categories'])
        for col in list(self.data.columns.values):
            # Solo para columnas categóricas
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                n_unique = self.data[col].nunique()
                if n_unique <= 10:
                    # Obtener las categorías únicas como lista
                    cats = self.data[col].unique().tolist()
                    categories.loc[col] = [cats]
                else:
                    categories.loc[col] = ['N/A (>10 categorías)']
            else:
                categories.loc[col] = ['N/A']
        
        # TAREA 3: Información estadística SOLO para columnas numéricas
        #Lista de valores máximos
        max_values = pd.DataFrame(columns=['Max_values'])
        for col in list(self.data.columns.values):
            if pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    max_values.loc[col] = [self.data[col].max()]
                except:
                    max_values.loc[col] = ['N/A']
            else:
                max_values.loc[col] = ['N/A']
        
        #Lista de valores mínimos
        min_values = pd.DataFrame(columns=['Min_values'])
        for col in list(self.data.columns.values):
            if pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    min_values.loc[col] = [self.data[col].min()]
                except:
                    min_values.loc[col] = ['N/A']
            else:
                min_values.loc[col] = ['N/A']
        
        #Lista de valores con la media
        mean_values = pd.DataFrame(columns=['Mean_values'])
        for col in list(self.data.columns.values):
            if pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    mean_values.loc[col] = [self.data[col].mean()]
                except:
                    mean_values.loc[col] = ['N/A']
            else:
                mean_values.loc[col] = ['N/A']
        
        #Lista de valores con su desviación estándar
        std_values = pd.DataFrame(columns=['Std_values'])
        for col in list(self.data.columns.values):
            if pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    std_values.loc[col] = [self.data[col].std()]
                except:
                    std_values.loc[col] = ['N/A']
            else:
                std_values.loc[col] = ['N/A']
           
        return (columns.join(data_dtypes).join(present_values).join(missing_values)
                .join(unique_values).join(is_categorical).join(categories)
                .join(max_values).join(min_values).join(mean_values).join(std_values))

    
  #  def categoricos_limpieza(self):
  #      for col in self.categoricos_columns:
  #          self.data_categoricos[col] = self.data_categoricos[col].apply(remove_punctuation)
            
    # remover signos de puntuación
    @staticmethod
    def remove_punctuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x

    

