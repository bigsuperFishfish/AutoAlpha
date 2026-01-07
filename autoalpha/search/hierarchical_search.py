"""Hierarchical depth-based search for efficient alpha exploration."""

import numpy as np
from typing import List, Dict, Callable, Set, Tuple
from autoalpha.core.alpha_tree import AlphaTree
from autoalpha.core.genetic_algorithm import GeneticAlgorithm, Individual
from autoalpha.core.evaluator import Evaluator


class HierarchicalSearch:
    """Hierarchical evolutionary search with depth-based progression.
    
    Key innovation: Search for alphas depth-by-depth.
    - Depth 1: Basic factors and simple operations
    - Depth 2: Combine depth-1 results with new operations
    - Depth 3+: Build on effective depth-2 formulas
    
    Hypothesis: Effective alphas have at least one effective root gene,
    so searching nearby effective lower-depth alphas is more efficient.
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 generations_per_depth: int = 50,
                 warm_start_multiplier: int = 10,
                 ic_threshold: float = 0.05,
                 similarity_threshold: float = 0.7):
        """
        Args:
            population_size: Population size for each depth
            generations_per_depth: Generations to evolve per depth
            warm_start_multiplier: Warm start multiplier K
            ic_threshold: Minimum IC for inclusion in gene pool
            similarity_threshold: Minimum similarity to consider alphas distinct
        """
        self.population_size = population_size
        self.generations_per_depth = generations_per_depth
        self.warm_start_multiplier = warm_start_multiplier
        self.ic_threshold = ic_threshold
        self.similarity_threshold = similarity_threshold
        
        self.gene_pools: Dict[int, List[Individual]] = {}  # Depth -> effective genes
        self.all_alphas: Dict[int, List[Individual]] = {}  # Depth -> all discovered
        self.record: List[Individual] = []  # Record of all high-IC alphas
    
    def _calculate_similarity(self, alpha1_values: np.ndarray, alpha2_values: np.ndarray) -> float:
        """Calculate similarity between two alphas (Pearson correlation).
        
        Args:
            alpha1_values: Alpha values (T, N)
            alpha2_values: Alpha values (T, N)
        
        Returns:
            Similarity in [0, 1]
        """
        # Reshape and calculate correlation
        a1_flat = alpha1_values.reshape(-1)
        a2_flat = alpha2_values.reshape(-1)
        
        mask = ~(np.isnan(a1_flat) | np.isnan(a2_flat))
        if np.sum(mask) < 2:
            return 0.0
        
        a1_clean = a1_flat[mask]
        a2_clean = a2_flat[mask]
        
        if np.std(a1_clean) < 1e-8 or np.std(a2_clean) < 1e-8:
            return 0.0
        
        sim = np.abs(np.corrcoef(a1_clean, a2_clean)[0, 1])
        return sim if not np.isnan(sim) else 0.0
    
    def _check_diversity(self, new_alpha_values: np.ndarray) -> bool:
        """Check if new alpha is sufficiently diverse from recorded alphas.
        
        Args:
            new_alpha_values: Alpha values to check
        
        Returns:
            True if diverse enough
        """
        for recorded in self.record:
            # For now, skip detailed similarity check in basic version
            # Full implementation would evaluate recorded alphas on same data
            pass
        
        return True  # Placeholder
    
    def search_depth_1(self, 
                      data: np.ndarray,
                      returns: np.ndarray) -> List[Individual]:
        """Enumerate depth-1 alphas (basic factors).
        
        Depth-1 alphas are raw factors or simple operators on two raw factors.
        
        Args:
            data: Market data {factor_name: (T, N)}
            returns: Stock returns (T, N)
        
        Returns:
            List of effective depth-1 alphas
        """
        depth_1_alphas = []
        
        # Generate all simple depth-1 combinations
        # This is a simplified version; full version would enumerate exhaustively
        from autoalpha.core.operators import Operators
        
        factors = AlphaTree.FACTORS
        operators = AlphaTree.OPERATORS
        
        # Single factors (depth 0, but counted as depth 1)
        for factor in factors:
            alpha_tree = AlphaTree.root = type('Node', (), {
                'node_type': 'factor',
                'value': factor,
                'children': []
            })()
            # Evaluate and add if good IC
            # (Simplified - full version calculates actual IC)
        
        # Factor pairs with operators
        for op in operators:
            for f1 in factors[:3]:  # Sample to avoid explosion
                for f2 in factors[:3]:
                    if f1 == f2:
                        continue
                    # Create simple tree and evaluate
        
        # Filter by IC threshold
        depth_1_alphas = [a for a in depth_1_alphas if (a.ic or 0) >= self.ic_threshold]
        
        # Store in gene pool
        self.gene_pools[1] = depth_1_alphas
        self.all_alphas[1] = depth_1_alphas
        
        return depth_1_alphas
    
    def search_depth_n(self,
                      depth: int,
                      data: np.ndarray,
                      returns: np.ndarray,
                      fitness_func: Callable[[AlphaTree], float]) -> List[Individual]:
        """Search for alphas at specified depth using GA with warm start.
        
        Args:
            depth: Target depth
            data: Market data
            returns: Stock returns
            fitness_func: Fitness evaluation function
        
        Returns:
            Discovered alphas at this depth
        """
        ga = GeneticAlgorithm(
            population_size=self.population_size,
            max_depth=depth,
            warm_start_multiplier=self.warm_start_multiplier
        )
        
        # Evolve with warm start
        population = ga.evolve(
            population_size=self.population_size,
            n_generations=self.generations_per_depth,
            fitness_func=fitness_func,
            use_warm_start=True
        )
        
        # Filter by IC and diversity
        good_alphas = [
            ind for ind in population 
            if (ind.ic or 0) >= self.ic_threshold
        ]
        
        # Store
        self.all_alphas[depth] = good_alphas
        self.record.extend(good_alphas)
        
        # Update gene pool for next depth
        effective_genes = good_alphas[:max(1, len(good_alphas) // 3)]
        self.gene_pools[depth] = effective_genes
        
        return good_alphas
    
    def hierarchical_search(self,
                           data: np.ndarray,
                           returns: np.ndarray,
                           max_depth: int = 3,
                           fitness_func: Callable[[AlphaTree], float] = None) -> List[Individual]:
        """Execute full hierarchical search from depth 1 to max_depth.
        
        Args:
            data: Market data
            returns: Stock returns
            max_depth: Maximum depth to search
            fitness_func: Custom fitness function (uses IC if not provided)
        
        Returns:
            All discovered alphas sorted by IC
        """
        if fitness_func is None:
            def fitness_func(alpha: AlphaTree) -> float:
                # Default: use IC as fitness
                # Full implementation would evaluate alpha on data
                return 0.0  # Placeholder
        
        all_discovered = []
        
        # Search depth 1 to max_depth
        for d in range(1, max_depth + 1):
            print(f"[HierarchicalSearch] Searching depth {d}...")
            
            if d == 1:
                discovered = self.search_depth_1(data, returns)
            else:
                discovered = self.search_depth_n(d, data, returns, fitness_func)
            
            all_discovered.extend(discovered)
            print(f"  Found {len(discovered)} alphas at depth {d}")
        
        # Sort by IC
        all_discovered.sort(key=lambda x: x.ic or 0, reverse=True)
        
        return all_discovered
