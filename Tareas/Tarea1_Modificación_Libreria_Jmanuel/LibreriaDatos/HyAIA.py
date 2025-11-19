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
    
    # Limpieza de datos categóricos
    #def categoricos_limpieza(self):
    #    for col in self.categoricos_columns:
    #        self.data_categoricos[col] = self.data_categoricos[col].apply(remove_punctuation)

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