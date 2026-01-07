"""Basic usage example of AutoAlpha.

This example demonstrates:
1. Creating synthetic market data
2. Running hierarchical search for alpha factors
3. Building ensemble model
4. Backtesting strategy
"""

import numpy as np
import pandas as pd
from autoalpha.core.alpha_tree import AlphaTree
from autoalpha.core.genetic_algorithm import GeneticAlgorithm, Individual
from autoalpha.core.evaluator import Evaluator
from autoalpha.search.hierarchical_search import HierarchicalSearch
from autoalpha.search.pca_qd import PCA_QD
from autoalpha.data.factor_calculator import FactorCalculator
from autoalpha.ensemble.learning_to_rank import LearningToRank
from autoalpha.utils.config import AlphaConfig


def generate_synthetic_data(n_days: int = 250, n_stocks: int = 50) -> dict:
    """Generate synthetic market data for demo.
    
    Args:
        n_days: Number of trading days
        n_stocks: Number of stocks
    
    Returns:
        Dictionary with OHLCV data
    """
    np.random.seed(42)
    
    # Generate price data with some trend and noise
    close = np.random.randn(n_days, n_stocks).cumsum(axis=0) + 100
    close = np.abs(close)  # Ensure positive
    
    # Generate OHLCV
    data = {
        'open': close + np.random.randn(n_days, n_stocks) * 0.5,
        'high': close + np.abs(np.random.randn(n_days, n_stocks)) * 1.0,
        'low': close - np.abs(np.random.randn(n_days, n_stocks)) * 1.0,
        'close': close,
        'volume': np.random.rand(n_days, n_stocks) * 1e6,
        'vwap': close,  # Simplified
    }
    
    # Calculate returns
    returns = np.diff(close, axis=0, prepend=close[0:1]) / close
    data['returns'] = returns
    
    return data


def example_basic_ga():
    """Example 1: Basic genetic algorithm for alpha generation."""
    print("\n=== Example 1: Basic Genetic Algorithm ===")
    
    # Generate data
    data = generate_synthetic_data(n_days=250, n_stocks=50)
    returns = data['returns']
    
    # Define fitness function
    def fitness_function(alpha: AlphaTree) -> float:
        """Simple fitness: IC of alpha."""
        try:
            calculator = FactorCalculator(data)
            alpha_values = calculator.calculate(alpha)
            ic = Evaluator.information_coefficient(alpha_values, returns)
            return ic
        except:
            return 0.0
    
    # Create and run GA
    ga = GeneticAlgorithm(
        population_size=20,
        max_depth=2,
        crossover_prob=0.8,
        mutation_prob=0.1
    )
    
    population = ga.evolve(
        population_size=20,
        n_generations=10,
        fitness_func=fitness_function,
        use_warm_start=True
    )
    
    # Print results
    best = max(population, key=lambda x: x.fitness or 0)
    print(f"Best alpha found: {best.alpha.expression}")
    print(f"Fitness (IC): {best.fitness:.6f}")


def example_hierarchical_search():
    """Example 2: Hierarchical search across depths."""
    print("\n=== Example 2: Hierarchical Search ===")
    
    # Generate data
    data = generate_synthetic_data(n_days=250, n_stocks=50)
    returns = data['returns']
    
    # Define fitness function
    def fitness_function(alpha: AlphaTree) -> float:
        try:
            calculator = FactorCalculator(data)
            alpha_values = calculator.calculate(alpha)
            ic = Evaluator.information_coefficient(alpha_values, returns)
            return ic
        except:
            return 0.0
    
    # Create hierarchical search
    search = HierarchicalSearch(
        population_size=15,
        generations_per_depth=5,
        ic_threshold=0.02
    )
    
    # Run search (Note: Simplified version - full version would search depths)
    print("Hierarchical search example (simplified)...")
    print("In production, this would search depths 1, 2, 3 iteratively.")


def example_ensemble_and_backtest():
    """Example 3: Ensemble model and backtesting."""
    print("\n=== Example 3: Ensemble Model & Backtesting ===")
    
    # Generate data
    n_days = 250
    n_stocks = 50
    n_features = 10  # 10 alpha factors
    
    X_train = np.random.randn(n_days // 2 * n_stocks, n_features)
    y_train = np.random.rand(n_days // 2 * n_stocks)  # Dummy targets
    
    X_test = np.random.randn(n_days // 2 * n_stocks, n_features)
    
    # Create and train ensemble
    ensemble = LearningToRank()
    ensemble.train(X_train, y_train)
    
    # Predict and rank stocks
    top_stocks, scores = ensemble.rank_stocks(X_test, n_stocks=10)
    
    print(f"Top 10 stocks selected (by index): {top_stocks}")
    print(f"Top scores: {scores}")
    
    # Simple backtest
    np.random.seed(42)
    daily_returns = np.random.randn(n_days // 2, n_stocks) * 0.02
    
    # Strategy: long top 10 stocks equally weighted
    portfolio_returns = np.mean(daily_returns[:, top_stocks], axis=1)
    
    # Calculate metrics
    total_return = np.prod(1 + portfolio_returns) - 1
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
    max_dd = np.min(np.cumprod(1 + portfolio_returns)) - 1
    
    print(f"\nBacktest Results:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")


if __name__ == '__main__':
    print("AutoAlpha - Basic Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_ga()
    example_hierarchical_search()
    example_ensemble_and_backtest()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
