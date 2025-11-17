import pandas as pd
import numpy as np


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
    # Método para obtener columnas y dataframe cuantitativos
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

    # ----------------------------
    # Utilidades internas
    # ----------------------------
    def _resolve_columns(self, columns):
        """
        Si columns es None, aplica a columnas categóricas no numéricas.
        Acepta string, lista de strings o None.
        """
        if columns is None:
            return list(self.data.select_dtypes(exclude='number').columns)
        if isinstance(columns, str):
            return [columns]
        return list(columns)

    def _apply_str_op(self, columns, op, inplace=True):
        """
        Aplica una operación de string (lambda s: ...) sobre columnas dadas.
        Convierte temporalmente a dtype 'string' para evitar errores con NaN/objetos.
        """
        cols = self._resolve_columns(columns)
        df = self.data if inplace else self.data.copy()
        for c in cols:
            # Solo operamos si la columna existe
            if c not in df.columns:
                continue
            # Convertimos a dtype de strings que tolera NaN
            s = df[c].astype("string")
            df[c] = op(s)
        if not inplace:
            return df
        # refrescar vistas internas si fue inplace
        self._refresh_views()
        return self

    def _refresh_views(self):
        self.data_binarios, self.binarios_columns = self.get_binarios()
        self.data_cuantitativos, self.cuantitativos_columns = self.get_cuantitativos()
        self.data_categoricos, self.categoricos_columns = self.get_categoricos()

    # ----------------------------
    # Limpieza de texto (vectorizada)
    # ----------------------------
    def text_remove_whitespace(self, columns=None, inplace=True):
        """
        Quita espacios extra internos y en extremos: '  Hola   mundo  ' -> 'Hola mundo'
        """
        return self._apply_str_op(
            columns,
            op=lambda s: s.str.split().str.join(" "),
            inplace=inplace
        )

    def text_lower(self, columns=None, inplace=True):
        """Convierte a minúsculas."""
        return self._apply_str_op(columns, op=lambda s: s.str.lower(), inplace=inplace)

    def text_upper(self, columns=None, inplace=True):
        """Convierte a MAYÚSCULAS."""
        return self._apply_str_op(columns, op=lambda s: s.str.upper(), inplace=inplace)

    def text_capitalize(self, columns=None, inplace=True):
        """Primera letra en mayúscula, resto minúsculas (tipo oración)."""
        return self._apply_str_op(columns, op=lambda s: s.str.capitalize(), inplace=inplace)

    def text_replace(self, to_replace, replacement, columns=None, inplace=True, regex=False):
        """
        Reemplaza texto exacto (por defecto) o por regex si regex=True.
        """
        return self._apply_str_op(
            columns,
            op=lambda s: s.str.replace(to_replace, replacement, regex=regex),
            inplace=inplace
        )

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

        # Información estadística
        #Lista de valores máximos
        max_values = pd.DataFrame(columns=['Max_values'])

        for col in list(self.data.columns.values):
            try:
                max_values.loc[col] = [self.data[col].max()]
            except:
                max_values.loc[col] = ['N/A']
                pass
        #Lista de valores mínimos
        min_values = pd.DataFrame(columns=['Min_values'])
        for col in list(self.data.columns.values):
            try:
                min_values.loc[col] = [self.data[col].min()]
            except:
                min_values.loc[col] = ['N/A']
                pass
    #Lista de valores con su desviación estandar
    
    #Lista de valores con los percentiles
    
    #Lista de valores con la media
       
        return columns.join(data_dtypes).join(present_values).join(missing_values).join(unique_values).join(max_values).join(min_values)
    # ----------------------------
    # Metodos estáticos de limpieza de texto
    # ----------------------------
    @staticmethod
    def remove_whitespace(x):
        try:
            return " ".join(str(x).split())
        except Exception:
            return x
    @staticmethod
    def lower_text(x):
        try:
            return str(x).lower()
        except Exception:
            return x

    @staticmethod
    def upper_text(x):
        try:
            return str(x).upper()
        except Exception:
            return x
    @staticmethod
    def capitalize_text(x):
        try:
            return str(x).capitalize()
        except Exception:
            return x

    @staticmethod
    def replace_text(x, to_replace, replacement):
        try:
            return str(x).replace(to_replace, replacement)
        except Exception:
            return x
    @staticmethod
    def remove_digits(x):
        try:
            return ''.join(ch for ch in str(x) if ch not in string.digits)
        except Exception:
            print(f'{x} no es una cadena de caracteres')
            return x

    @staticmethod
    def remove_punctuation(x):
        try:
            return ''.join(ch for ch in str(x) if ch not in string.punctuation)
        except Exception:
            print(f'{x} no es una cadena de caracteres')
            return x
