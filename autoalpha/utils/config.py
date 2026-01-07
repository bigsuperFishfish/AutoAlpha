"""Configuration and hyperparameters for AutoAlpha."""

from dataclasses import dataclass
from typing import List


@dataclass
class AlphaConfig:
    """Configuration for AutoAlpha algorithm."""
    
    # Genetic Algorithm Parameters
    population_size: int = 100
    max_depth: int = 3
    generations_per_depth: int = 50
    warm_start_multiplier: int = 10
    
    # Operators and Search
    crossover_prob: float = 0.8
    mutation_prob: float = 0.1
    
    # Evaluation
    holding_period: int = 1  # days
    ic_threshold: float = 0.05
    similarity_threshold: float = 0.7
    pca_similarity_threshold: float = 0.9
    
    # Ensemble
    top_k_alphas: int = 150
    long_n_stocks: int = 10
    transaction_cost: float = 0.003  # 0.3%
    risk_free_rate: float = 0.0
    trading_days_per_year: int = 252
    
    # Technical Indicators
    technical_windows: List[int] = None
    
    def __post_init__(self):
        if self.technical_windows is None:
            self.technical_windows = [5, 10, 20, 60]
