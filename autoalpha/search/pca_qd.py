"""PCA-QD: Quality Diversity search using PCA similarity.

Instead of calculating full Pearson correlation (O(npT)), we use
PCA-similarity based on first principal component (O(pT)).
"""

import numpy as np
from typing import List, Optional
from sklearn.decomposition import PCA
from autoalpha.core.alpha_tree import AlphaTree
from autoalpha.core.genetic_algorithm import Individual


class PCA_QD:
    """Quality Diversity search with PCA-based similarity approximation."""
    
    def __init__(self,
                 similarity_threshold: float = 0.9,
                 pca_components: int = 1):
        """
        Args:
            similarity_threshold: Threshold for PCA-similarity penalty
            pca_components: Number of PCA components to use
        """
        self.similarity_threshold = similarity_threshold
        self.pca_components = pca_components
        self.record: List[np.ndarray] = []  # Record of alpha PCA projections
    
    def calculate_pca_similarity(self,
                                alpha_values: np.ndarray,
                                reference_values: np.ndarray) -> float:
        """Calculate PCA-similarity between two alphas.
        
        Uses first principal component of alpha values, then calculates
        Pearson correlation with reference's first PC.
        
        Args:
            alpha_values: Alpha values (T, N)
            reference_values: Reference alpha values (T, N)
        
        Returns:
            PCA-similarity in [0, 1]
        """
        try:
            # Reshape: (T, N) -> (T, N) for PCA
            T, N = alpha_values.shape
            
            # Calculate first PC for alpha
            alpha_clean = np.nan_to_num(alpha_values, nan=0.0)
            if np.std(alpha_clean) < 1e-8:
                return 0.0
            
            pca_alpha = PCA(n_components=1)
            pc_alpha = pca_alpha.fit_transform(alpha_clean)  # (T, 1)
            pc_alpha = pc_alpha.reshape(-1)
            
            # Calculate first PC for reference
            ref_clean = np.nan_to_num(reference_values, nan=0.0)
            if np.std(ref_clean) < 1e-8:
                return 0.0
            
            pca_ref = PCA(n_components=1)
            pc_ref = pca_ref.fit_transform(ref_clean)  # (T, 1)
            pc_ref = pc_ref.reshape(-1)
            
            # Correlation between principal components
            if np.std(pc_alpha) < 1e-8 or np.std(pc_ref) < 1e-8:
                return 0.0
            
            sim = np.abs(np.corrcoef(pc_alpha, pc_ref)[0, 1])
            return float(sim) if not np.isnan(sim) else 0.0
        
        except Exception as e:
            print(f"PCA calculation error: {e}")
            return 0.0
    
    def check_diversity(self,
                       new_alpha_values: np.ndarray,
                       recorded_alphas: List[np.ndarray]) -> bool:
        """Check if new alpha is sufficiently diverse.
        
        Args:
            new_alpha_values: Alpha values to check (T, N)
            recorded_alphas: List of recorded alpha values
        
        Returns:
            True if diverse (below similarity threshold)
        """
        if not recorded_alphas:
            return True
        
        for recorded in recorded_alphas:
            sim = self.calculate_pca_similarity(new_alpha_values, recorded)
            if sim > self.similarity_threshold:
                return False  # Too similar
        
        return True  # Diverse enough
    
    def apply_diversity_penalty(self,
                               fitness: float,
                               new_alpha_values: np.ndarray,
                               recorded_alphas: List[np.ndarray]) -> float:
        """Apply diversity penalty to fitness if too similar to recorded alphas.
        
        Args:
            fitness: Original fitness value (IC)
            new_alpha_values: New alpha values
            recorded_alphas: List of recorded alphas
        
        Returns:
            Adjusted fitness (0 if too similar, original otherwise)
        """
        if not self.check_diversity(new_alpha_values, recorded_alphas):
            return 0.0  # Penalize to 0
        
        return fitness
    
    def record_alpha(self, alpha_values: np.ndarray):
        """Record alpha for future diversity checks.
        
        Args:
            alpha_values: Alpha values (T, N)
        """
        self.record.append(alpha_values.copy())
    
    def get_diversity_info(self) -> dict:
        """Get current diversity information.
        
        Returns:
            Dict with diversity statistics
        """
        return {
            'num_recorded': len(self.record),
            'similarity_threshold': self.similarity_threshold,
            'pca_components': self.pca_components
        }
