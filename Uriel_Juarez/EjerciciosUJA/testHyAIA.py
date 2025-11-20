#import Package_HyAIA.HyAIA as HyAIA

import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))    
csv_path = os.path.join(base, "countries.csv")  
print(csv_path)
df = pd.read_csv(csv_path, sep=';')
print(df.head())