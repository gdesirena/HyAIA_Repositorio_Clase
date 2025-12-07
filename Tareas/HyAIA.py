import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import deque

class HyAIA:
    def __init__(self, df):
        self.data = df
        self.columns = df.columns
        self.data_binarios, self.binarios_columns = self.get_binarios()
        self.data_cuantitativos, self.cuantitativos_columns = self.get_cuantitativos()
        self.data_categoricos, self.categoricos_columns = self.get_categoricos()
        self.df_dqr = self.get_dqr()


class LOF:
    """
    Implementación educativa del algoritmo Local Outlier Factor (LOF)
    sin usar sklearn.
    """

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        """
        Ajusta el modelo calculando:
        - k-distance
        - vecinos más cercanos
        - densidad local
        - LOF score
        """

        # Convertimos a matriz numpy
        self.X = np.array(X)

        # Calculamos vecinos
        knn = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)
        distances, indices = knn.kneighbors(self.X)

        # Quitamos distancia 0 (el mismo punto)
        self.distances = distances[:, 1:]
        self.indices = indices[:, 1:]

        # k-distance
        self.k_distance = self.distances[:, -1]

        # Reachability distance
        reach_dist = np.maximum(
            self.distances,
            self.k_distance[self.indices]
        )
        self.reachability = reach_dist

        # Local reachability density (LRD)
        self.lrd = 1 / (np.mean(self.reachability, axis=1) + 1e-10)

        # LOF score
        self.lof_score = np.zeros(len(self.X))
        for i in range(len(self.X)):
            neighbors = self.indices[i]
            self.lof_score[i] = np.mean(self.lrd[neighbors] / self.lrd[i])

        return self

    def predict(self, threshold=1.5):
        """
        Regresa etiquetas:
        - 1  → normal
        - -1 → outlier
        """
        return np.where(self.lof_score > threshold, -1, 1)

    def fit_predict(self, X, threshold=1.5):
        self.fit(X)
        return self.predict(threshold)

class DBSCAN:
    """
    Implementación educativa de DBSCAN:
    
    Parámetros:
    - eps: radio de vecindad
    - min_samples: mínimo número de puntos para formar un cluster
    
    Etiquetas generadas:
    - -1 → ruido (outlier)
    - 0, 1, 2... → número de cluster
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        
    def fit(self, X):
        self.X = np.array(X)
        self.labels_ = np.full(len(self.X), -1)   # todos como ruido
        visited = np.zeros(len(self.X), dtype=bool)
        
        cluster_id = 0
        
        for i in range(len(self.X)):
            if visited[i]:
                continue
                
            visited[i] = True
            neighbors = self._region_query(i)
            
            # Si no es "core point"
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                # Expandir cluster
                self._expand_cluster(i, neighbors, cluster_id, visited)
                cluster_id += 1
                
        self.n_clusters_ = cluster_id
        return self

    def _expand_cluster(self, point_idx, neighbors, cluster_id, visited):
        queue = deque(neighbors)
        self.labels_[point_idx] = cluster_id
        
        while queue:
            idx = queue.popleft()
            
            if not visited[idx]:
                visited[idx] = True
                new_neighbors = self._region_query(idx)
                
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)
            
            # asigna cluster si aún es ruido
            if self.labels_[idx] == -1:
                self.labels_[idx] = cluster_id

    def _region_query(self, index):
        distances = np.linalg.norm(self.X - self.X[index], axis=1)
        return np.where(distances <= self.eps)[0]

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_



