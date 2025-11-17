import Package_HyAIA.HyAIA as HyAIA

import numpy as np
import pandas as pd



df_bank = pd.read_csv('Introduccion/Data/bank.csv')
df_bank.head()


hy = HyAIA.HyAIA(df_bank)
print("Columnas Binarias:", hy.binarios_columns)
print("Columnas Binarias:", hy.categoricos_columns)