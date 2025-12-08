
import pandas as pd
import numpy as np
import string
from sklearn.neighbors import NearestNeighbors

class HyAIA:
    def __init__(self, df):
        self.data = df
        self.columns = df.columns
        self.df_dqr = self.get_dqr()
        
    def get_dqr(self):
        # Tu código original del DQR (resumido para ahorrar espacio)
        columns = pd.DataFrame(list(self.data.columns.values), columns=['Columns_Names'], index=list(self.data.columns.values))
        data_dtypes = pd.DataFrame(self.data.dtypes, columns=['Dtypes'])
        present_values = pd.DataFrame(self.data.count(), columns=['Present_values'])
        missing_values = pd.DataFrame(self.data.isnull().sum(), columns=['Missing_values'])
        unique_values = pd.DataFrame(columns=['Unique_values'])
        for col in list(self.data.columns.values):
            unique_values.loc[col] = [self.data[col].nunique()]
        return columns.join(data_dtypes).join(present_values).join(missing_values).join(unique_values)

# --- CLASE 1: IMPLEMENTACIÓN MANUAL DE LOF ---
class LOF_Manual:
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.lof_scores_ = None
        self.negative_outlier_factor_ = None

    def fit_predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        
        # 1. Encontrar k-vecinos y sus distancias
        # Usamos NearestNeighbors para la búsqueda eficiente, pero la lógica LOF es manual
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Eliminamos el punto mismo (distancia 0)
        dist_k = distances[:, 1:] 
        neighbors_indices = indices[:, 1:]
        
        # K-distance: distancia al k-ésimo vecino
        k_distance = dist_k[:, -1]
        
        # 2. Calcular Reachability Distance
        # reach_dist(A, B) = max(k_distance(B), dist(A,B))
        reach_dist_array = np.zeros((n_samples, self.n_neighbors))
        
        for i in range(n_samples):
            for j in range(self.n_neighbors):
                neighbor_idx = neighbors_indices[i, j]
                dist_A_B = dist_k[i, j]
                k_dist_B = k_distance[neighbor_idx]
                reach_dist_array[i, j] = max(k_dist_B, dist_A_B)
        
        # 3. Local Reachability Density (LRD)
        # Inverso del promedio de la reachability distance de los vecinos
        lrd = 1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)
        
        # 4. Local Outlier Factor (LOF)
        # Promedio del LRD de los vecinos / LRD del punto
        lof_scores = np.zeros(n_samples)
        for i in range(n_samples):
            lrd_neighbors = lrd[neighbors_indices[i]]
            lof_scores[i] = np.mean(lrd_neighbors) / lrd[i]
            
        self.lof_scores_ = lof_scores
        self.negative_outlier_factor_ = -lof_scores # Para compatibilidad con sklearn
        
        # Definimos threshold simple (ej. > 1.5 es outlier) o retornamos scores
        # Aquí retornamos -1 para outliers (score > 1.5) y 1 para inliers
        return np.where(lof_scores > 1.5, -1, 1)

# --- CLASE 2: IMPLEMENTACIÓN MANUAL DE DBSCAN ---
class DBSCAN_Manual:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit_predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        labels = np.full(n_samples, 0) # 0: unvisited
        cluster_id = 0
        
        # Buscador de vecinos por radio
        nbrs = NearestNeighbors(radius=self.eps).fit(X)
        adj_matrix = nbrs.radius_neighbors(X, return_distance=False)
        
        for i in range(n_samples):
            if labels[i] != 0:
                continue
            
            neighbors = adj_matrix[i]
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1 # Ruido (Noise)
            else:
                cluster_id += 1
                self._expand_cluster(X, labels, i, neighbors, cluster_id, adj_matrix)
                
        self.labels_ = labels
        return labels

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id, adj_matrix):
        labels[point_idx] = cluster_id
        
        # Usamos una lista para iterar (similar a una cola)
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if labels[neighbor_idx] == -1: # Era ruido, ahora es borde
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0: # No visitado
                labels[neighbor_idx] = cluster_id
                
                new_neighbors = adj_matrix[neighbor_idx]
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            i += 1
