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
        
    

    # Columnas binarias (nunique == 2)
    def get_binarios(self):
        col_bin = []
        for col in self.data.columns:
            if self.data[col].nunique() == 2:
                col_bin.append(col)
        return self.data[col_bin], col_bin
        
    # Columnas numéricas
    def get_cuantitativos(self):
        col_cuantitativas = self.data.select_dtypes(include='number').columns
        return self.data[col_cuantitativas], col_cuantitativas
        
    # Columnas categóricas (no numéricas)
    def get_categoricos(self):
        col_categoricos = self.data.select_dtypes(exclude='number').columns
        col_cat = []
        for col in col_categoricos:
            if self.data[col].nunique() > 2:
                col_cat.append(col)
        return self.data[col_cat], col_cat
    
   
    def get_dqr(self):
        columns = pd.DataFrame(list(self.data.columns.values), 
                               columns=['Columns_Names'], 
                               index=list(self.data.columns.values))
        
        data_dtypes = pd.DataFrame(self.data.dtypes, columns=['Dtypes'])
        present_values = pd.DataFrame(self.data.count(), columns=['Present_values'])
        missing_values = pd.DataFrame(self.data.isnull().sum(), columns=['Missing_values'])
        
        unique_values = pd.DataFrame(columns=['Unique_values'])
        for col in list(self.data.columns.values):
            unique_values.loc[col] = [self.data[col].nunique()]
        
        max_values = pd.DataFrame(columns=['Max_values'])
        for col in list(self.data.columns.values):
            try:
                max_values.loc[col] = [self.data[col].max()]
            except:
                max_values.loc[col] = ['N/A']
        
        min_values = pd.DataFrame(columns=['Min_values'])
        for col in list(self.data.columns.values):
            try:
                min_values.loc[col] = [self.data[col].min()]
            except:
                min_values.loc[col] = ['N/A']
                
        return (columns.join(data_dtypes)
                      .join(present_values)
                      .join(missing_values)
                      .join(unique_values)
                      .join(max_values)
                      .join(min_values))


    @staticmethod
    def remove_punctuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            print(f'{x} no es una cadena de caracteres')
        return x

    def dqr(self):
        """
        Reporte de Calidad de Datos (DQR)
        Requisitos:
        - Identificar si una columna es categórica
        - Agregar categorías si tiene <= 10 valores únicos
        - Calcular max, min, mean, std solo si es numérica
        """

        df = self.data

        # Tipo de dato
        tipos = df.dtypes

        # Nulos
        nulos = df.isnull().sum()
        pct_nulos = round((df.isnull().sum() / len(df)) * 100, 2)

        # 1) Identificar columnas categóricas
        is_categorical = df.apply(lambda col: col.dtype == 'object' or col.dtype.name == 'category')

        # 2) Categorías (solo si ≤ 10 valores únicos)
        categorias = []
        for col in df.columns:
            if is_categorical[col]:
                uniques = df[col].dropna().unique()
                if len(uniques) <= 10:
                    categorias.append(list(uniques))
                else:
                    categorias.append(None)
            else:
                categorias.append(None)

        # 3) Estadísticos solo para numéricas
        max_vals = []
        min_vals = []
        mean_vals = []
        std_vals = []

        for col in df.columns:
            if not is_categorical[col]:  # si no es categórica = numérica
                max_vals.append(df[col].max())
                min_vals.append(df[col].min())
                mean_vals.append(df[col].mean())
                std_vals.append(df[col].std())
            else:
                max_vals.append(None)
                min_vals.append(None)
                mean_vals.append(None)
                std_vals.append(None)

        # Construcción del DataFrame final
        dqr_df = pd.DataFrame({
            'tipo_dato': tipos,
            'nulos_total': nulos,
            'porcentaje_nulos': pct_nulos,
            'is_categorical': is_categorical,
            'categories': categorias,
            'max': max_vals,
            'min': min_vals,
            'mean': mean_vals,
            'std': std_vals
        })

        return dqr_df
