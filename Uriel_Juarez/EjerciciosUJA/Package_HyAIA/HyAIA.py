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

    # ------------------------------------------------------------------
    # Métodos para clasificaciones 
    # ------------------------------------------------------------------

    def get_binarios(self):
        """Columnas con exactamente 2 valores únicos."""
        col_bin = []
        for col in self.data.columns:
            if self.data[col].nunique() == 2:
                col_bin.append(col)
        return self.data[col_bin], col_bin

    def get_cuantitativos(self):
        """Columnas numericas detectadas por pandas."""
        col_cuantitativas = self.data.select_dtypes(include="number").columns
        return self.data[col_cuantitativas], col_cuantitativas

    def get_categoricos(self):
        """Columnas categoricas (dtype no numérico y más de 2 categorías)."""
        col_categoricos = self.data.select_dtypes(exclude="number").columns
        col_cat = []
        for col in col_categoricos:
            if self.data[col].nunique() > 2:
                col_cat.append(col)
        return self.data[col_cat], col_cat

    # ------------------------------------------------------------------
    # Data Quality Report
    # ------------------------------------------------------------------
    def get_dqr(self):

        # ------------------------ Columnas ------------------------
        columns = pd.DataFrame(
            list(self.data.columns.values),
            columns=["Columns_Names"],
            index=list(self.data.columns.values),
        )

        # ------------------------ Dtypes ------------------------
        data_dtypes = pd.DataFrame(self.data.dtypes, columns=["Dtypes"])

        # ------------------------ Valores presentes ------------------------
        present_values = pd.DataFrame(self.data.count(), columns=["Present_values"])

        # ------------------------ Valores nulos ------------------------
        missing_values = pd.DataFrame(self.data.isnull().sum(), columns=["Missing_values"])

        # ------------------------ Valores únicos ------------------------
        unique_values = pd.DataFrame(columns=["Unique_values"])
        for col in self.data.columns:
            unique_values.loc[col] = [self.data[col].nunique()]

        # ------------------------ Is_categorical ------------------------
        is_categorical = pd.DataFrame(
            [col in self.categoricos_columns for col in self.data.columns],
            index=self.data.columns,
            columns=["Is_categorical"],
        )

        # ------------------------------------------------------------------
        # Estadísticos solo para columnas numéricas
        # ------------------------------------------------------------------
        max_values = pd.DataFrame(index=self.data.columns, columns=["Max_values"])
        min_values = pd.DataFrame(index=self.data.columns, columns=["Min_values"])
        mean_values = pd.DataFrame(index=self.data.columns, columns=["Mean_values"])
        std_values = pd.DataFrame(index=self.data.columns, columns=["Std_values"])

        for col in self.data.columns:

            # Intento de conversión numérica segura
            col_numeric = pd.to_numeric(self.data[col], errors="coerce")

            # Definir si es numérica por ratio de valores válidos
            numeric_ratio = col_numeric.notna().mean()

            if numeric_ratio >= 0.8:  # Consideramos numérica
                max_values.loc[col] = col_numeric.max()
                min_values.loc[col] = col_numeric.min()
                mean_values.loc[col] = col_numeric.mean()
                std_values.loc[col] = col_numeric.std()
            else:  # No numérica
                max_values.loc[col] = np.nan
                min_values.loc[col] = np.nan
                mean_values.loc[col] = np.nan
                std_values.loc[col] = np.nan

        # ------------------------------------------------------------------
        # Categorías solo para columnas categóricas con <=10 valores únicos
        # ------------------------------------------------------------------
        categories = pd.DataFrame(index=self.data.columns, columns=["Categories"])

        for col in self.data.columns:
            if (col in self.categoricos_columns) and (
                unique_values.loc[col, "Unique_values"] <= 10
            ):
                cats = self.data[col].dropna().unique()
                categories.loc[col, "Categories"] = ", ".join(map(str, cats))
            else:
                categories.loc[col, "Categories"] = np.nan


        report = (
            columns.join(data_dtypes)
            .join(present_values)
            .join(missing_values)
            .join(unique_values)
            .join(is_categorical)
            .join(categories)
            .join(max_values)
            .join(min_values)
            .join(mean_values)
            .join(std_values)
        )

        return report

    # ------------------------------------------------------------------
    # Operaciones de limpieza de texto 
    # ------------------------------------------------------------------

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
            return "".join(ch for ch in str(x) if ch not in string.digits)
        except Exception:
            return x

    @staticmethod
    def remove_punctuation(x):
        try:
            return "".join(ch for ch in str(x) if ch not in string.punctuation)
        except Exception:
            return x
