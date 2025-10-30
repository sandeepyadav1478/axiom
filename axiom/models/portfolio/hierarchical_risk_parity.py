"""Hierarchical Risk Parity (HRP) - Lopez de Prado"""
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

class HierarchicalRiskParity:
    def optimize(self, returns_data):
        """HRP optimization using hierarchical clustering"""
        cov = np.cov(returns_data.T)
        corr = np.corrcoef(returns_data.T)
        
        # Distance matrix
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)
        
        # Hierarchical clustering
        link = linkage(squareform(dist), method='single')
        
        # Quasi-diagonalization
        sort_ix = self._quasi_diag(link)
        
        # Recursive bisection
        weights = self._recursive_bisection(cov, sort_ix)
        
        return weights
    
    def _quasi_diag(self, link):
        """Quasi-diagonalization"""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i+1)
            sort_ix = sort_ix.append(df0).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        
        return sort_ix.tolist()
    
    def _recursive_bisection(self, cov, sort_ix):
        """Recursive bisection for weights"""
        n = len(sort_ix)
        w = np.ones(n)
        
        def _bisect(items):
            if len(items) <= 1:
                return
            
            split = len(items) // 2
            left = items[:split]
            right = items[split:]
            
            # Calculate variances
            cov_left = cov[np.ix_(left, left)]
            cov_right = cov[np.ix_(right, right)]
            
            var_left = w[left].T @ cov_left @ w[left]
            var_right = w[right].T @ cov_right @ w[right]
            
            # Inverse variance weighting
            alpha = var_right / (var_left + var_right)
            
            w[left] *= alpha
            w[right] *= (1 - alpha)
            
            _bisect(left)
            _bisect(right)
        
        _bisect(list(range(n)))
        
        return w / w.sum()

if __name__ == "__main__":
    import pandas as pd
    returns = np.random.randn(100, 5) * 0.01
    hrp = HierarchicalRiskParity()
    w = hrp.optimize(returns)
    print(f"HRP: {w}")