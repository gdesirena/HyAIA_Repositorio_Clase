import pandas as pd
import numpy as np
import string
from scipy.spatial.distance import cdist

class HyAIA:
    def __init__(self, df):
        self.data = df
        self.columns = df.columns
        self.data_binarios, self.binarios_columns = self.get_binarios()
        self.data_cuantitativos, self.cuantitativos_columns = self.get_cuantitativos()
        self.data_categoricos, self.categoricos_columns = self.get_categoricos()
        self.df_dqr = self.get_dqr()
        
    def get_binarios(self):
        col_bin = []
        for col in self.data.columns:
            if self.data[col].nunique() == 2:
                col_bin.append(col)
        return self.data[col_bin], col_bin
        
    def get_cuantitativos(self):
        col_cuantitativas = self.data.select_dtypes(include='number').columns
        return self.data[col_cuantitativas], col_cuantitativas
        
    def get_categoricos(self):
        col_categoricos = self.data.select_dtypes(exclude='number').columns
        col_cat = []
        for col in col_categoricos:
            if self.data[col].nunique()>2:
                col_cat.append(col)
        return self.data[col_cat], col_cat
        
    def get_dqr(self):
        columns = pd.DataFrame(list(self.data.columns.values), columns=['Columns_Names'], 
                               index=list(self.data.columns.values))
        
        data_dtypes = pd.DataFrame(self.data.dtypes, columns=['Dtypes'])
        
        present_values = pd.DataFrame(self.data.count(), columns=['Present_values'])
        
        missing_values = pd.DataFrame(self.data.isnull().sum(), columns=['Missing_values'])
        
        unique_values = pd.DataFrame(columns=['Unique_values'])
        for col in list(self.data.columns.values):
            unique_values.loc[col] = [self.data[col].nunique()]
        
        is_categorical = pd.DataFrame(columns=['Is_categorical'])
        for col in list(self.data.columns.values):
            is_categorical.loc[col] = [not pd.api.types.is_numeric_dtype(self.data[col])]
        
        categories = pd.DataFrame(columns=['Categories'])
        for col in list(self.data.columns.values):
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                n_unique = self.data[col].nunique()
                if n_unique <= 10:
                    cats = self.data[col].unique().tolist()
                    categories.loc[col] = [cats]
                else:
                    categories.loc[col] = ['N/A (>10 categorías)']
            else:
                categories.loc[col] = ['N/A']
        
        max_values = pd.DataFrame(columns=['Max_values'])
        for col in list(self.data.columns.values):
            if pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    max_values.loc[col] = [self.data[col].max()]
                except:
                    max_values.loc[col] = ['N/A']
            else:
                max_values.loc[col] = ['N/A']
        
        min_values = pd.DataFrame(columns=['Min_values'])
        for col in list(self.data.columns.values):
            if pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    min_values.loc[col] = [self.data[col].min()]
                except:
                    min_values.loc[col] = ['N/A']
            else:
                min_values.loc[col] = ['N/A']
        
        mean_values = pd.DataFrame(columns=['Mean_values'])
        for col in list(self.data.columns.values):
            if pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    mean_values.loc[col] = [self.data[col].mean()]
                except:
                    mean_values.loc[col] = ['N/A']
            else:
                mean_values.loc[col] = ['N/A']
        
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

    @staticmethod
    def remove_punctuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x


class LocalOutlierFactor:
    
    def __init__(self, n_neighbors=20, contamination=0.1, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.lof_scores_ = None
        self.threshold_ = None
        
    def fit_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        n_samples = X.shape[0]
        distances = cdist(X, X, metric=self.metric)
        
        k = min(self.n_neighbors, n_samples - 1)
        neighbor_indices = np.argsort(distances, axis=1)[:, 1:k+1]
        
        k_distances = np.zeros(n_samples)
        for i in range(n_samples):
            k_distances[i] = distances[i, neighbor_indices[i, -1]]
        
        reach_dist = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j_idx, j in enumerate(neighbor_indices[i]):
                reach_dist[i, j_idx] = max(distances[i, j], k_distances[j])
        
        lrd = np.zeros(n_samples)
        for i in range(n_samples):
            mean_reach_dist = np.mean(reach_dist[i])
            if mean_reach_dist > 0:
                lrd[i] = 1.0 / mean_reach_dist
            else:
                lrd[i] = np.inf
        
        lof_scores = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = neighbor_indices[i]
            if lrd[i] == np.inf:
                lof_scores[i] = 1.0
            else:
                lrd_ratios = lrd[neighbors] / lrd[i]
                lof_scores[i] = np.mean(lrd_ratios)
        
        self.lof_scores_ = lof_scores
        threshold = np.percentile(lof_scores, 100 * (1 - self.contamination))
        self.threshold_ = threshold
        
        labels = np.where(lof_scores > threshold, -1, 1)
        return labels
    
    def get_lof_scores(self):
        return self.lof_scores_


class DBSCAN:
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        
    def fit_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        n_samples = X.shape[0]
        distances = cdist(X, X, metric=self.metric)
        
        labels = np.full(n_samples, -1)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_samples):
            if visited[i]:
                continue
                
            visited[i] = True
            neighbors = np.where(distances[i] <= self.eps)[0]
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                seed_set = list(neighbors)
                
                k = 0
                while k < len(seed_set):
                    q = seed_set[k]
                    
                    if not visited[q]:
                        visited[q] = True
                        neighbors_q = np.where(distances[q] <= self.eps)[0]
                        
                        if len(neighbors_q) >= self.min_samples:
                            for neighbor in neighbors_q:
                                if neighbor not in seed_set:
                                    seed_set.append(neighbor)
                    
                    if labels[q] == -1:
                        labels[q] = cluster_id
                    
                    k += 1
                
                cluster_id += 1
        
        self.labels_ = labels
        self.core_sample_indices_ = np.where(labels != -1)[0]
        return labels
    
    def get_outliers(self):
        return np.where(self.labels_ == -1)[0]
    
    def get_n_clusters(self):
        if self.labels_ is None:
            return 0
        return len(np.unique(self.labels_[self.labels_ != -1]))
