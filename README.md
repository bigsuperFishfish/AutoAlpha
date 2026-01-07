# AutoAlpha

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors in Quantitative Investment.

Based on **Zhang et al. (2020)**: "AutoAlpha: an Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors in Quantitative Investment"

## Overview

AutoAlpha is a genetic programming-based system that automatically discovers effective formulaic alpha factors from historical stock data. Unlike manual alpha factor design which requires domain expertise, AutoAlpha leverages evolutionary computation to:

- **Automatically generate** formulaic alphas from basic operators (+, -, *, /, min, max) and technical indicators
- **Efficiently search** large factor space using hierarchical structure and PCA-QD diversity search
- **Prevent overfitting** with out-of-sample testing and anti-premature convergence mechanisms
- **Generate portfolios** using ensemble learning-to-rank models (LightGBM + XGBoost)

## Key Features

### 1. Hierarchical Structure
- **Multi-level search** from depth 1 → depth 2 → depth 3 factors
- Effective root genes from lower depths feed into higher-depth exploration
- Significantly more efficient than vanilla genetic programming (e.g., gplearn)

### 2. PCA-QD Search
- **Quality Diversity** mechanism guides exploration away from already-discovered alphas
- **Reduced complexity**: O(pT) vs O(npT) using PCA-similarity approximation
- Finds diverse, non-correlated alpha factors (similarity < 0.7)

### 3. Convergence Prevention
- **Warm Start**: Initialize population with top-K individuals (K times population size)
- **Replacement Method**: Parent-offspring competition maintains genetic diversity
- **Gene pool**: Maintains effective genes across generations

### 4. Fitness Evaluation
- **Information Coefficient (IC)**: Measures alpha-return correlation
- **Sharpe Ratio**: Risk-adjusted performance metric
- **Out-of-sample testing**: Validates generalization

## Project Structure

```
AutoAlpha/
├── autoalpha/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── alpha_tree.py           # Tree representation of alphas
│   │   ├── genetic_algorithm.py    # Core GA implementation
│   │   ├── operators.py            # Mathematical operators and indicators
│   │   └── evaluator.py            # Fitness evaluation (IC, Sharpe)
│   ├── search/
│   │   ├── __init__.py
│   │   ├── hierarchical_search.py  # Hierarchical depth-based search
│   │   └── pca_qd.py              # PCA-QD diversity maintenance
│   ├── ensemble/
│   │   ├── __init__.py
│   │   └── learning_to_rank.py    # LightGBM/XGBoost ensemble
│   ├── data/
│   │   ├── __init__.py
│   │   └── factor_calculator.py    # Alpha expression evaluation
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration and hyperparameters
│       └── logger.py               # Logging utilities
├── examples/
│   └── basic_usage.py              # Example usage
├── tests/
│   └── test_alpha_generation.py    # Unit tests
├── requirements.txt                 # Dependencies
├── setup.py                        # Package setup
└── README.md
```

## Installation

```bash
git clone https://github.com/bigsuperFishfish/AutoAlpha.git
cd AutoAlpha
pip install -r requirements.txt
```

## Dependencies

- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `lightgbm` - Gradient boosting (ensemble)
- `xgboost` - Gradient boosting (ensemble)
- `deap` - Distributed Evolutionary Algorithms in Python (genetics framework)

## Quick Start

```python
import pandas as pd
from autoalpha import AutoAlpha
from autoalpha.utils.config import AlphaConfig

# Load market data
data = pd.read_csv('stock_data.csv')

# Configure algorithm
config = AlphaConfig(
    population_size=100,
    max_depth=3,
    generations=50,
    holding_period=1,  # 1-day holding
    ic_threshold=0.05,
    similarity_threshold=0.7
)

# Initialize and run
aa = AutoAlpha(config=config)
alphas = aa.search(data)

# Build ensemble model and backtest
model = aa.build_ensemble(alphas, data_train)
portfolio = aa.backtest(model, data_test)
```

## Algorithm Overview

### Step 1: Hierarchical Gene Pool Initialization
1. Enumerate all depth-1 alphas (single operators on raw factors)
2. Rank by IC and select top performers
3. Build gene pool for depth-2 search

### Step 2: Genetic Algorithm with PCA-QD
For each depth level:
1. **Warm Start**: Generate K × population_size candidates, select top 1/K by IC
2. **Crossover**: Combine genes at same tree depth
3. **Mutation**: Random subtree replacement
4. **PCA-QD Evaluation**: 
   - Calculate PCA-similarity with recorded alphas
   - Apply penalty if similarity > threshold
   - Keep diverse, high-IC alphas
5. **Parent-Offspring Replacement**: Only keep if fitness > parents

### Step 3: Ensemble Portfolio Generation
1. Use top-150 alphas (by IC) as features
2. Train LightGBM + XGBoost ranking models
3. Ensemble predictions → stock rankings
4. Long top-10 stocks each day

## Performance (Backtesting Results)

On CSI 800 Index (2017-09-01 to 2019-07-31):

| Metric | h=1 | h=5 |
|--------|-----|-----|
| Annualized Return | 90.0% | 28.0% |
| Sharpe Ratio | 3.39 | 1.20 |
| vs Market (CSI 800) | +98.2% AR | +34.0% AR |
| Discovered Alphas (IC>0.05) | 434 | 415 |

Comparison with baselines:
- **gplearn**: 61.8% AR (v.s. 90.0%)
- **Alpha101**: 29.5% AR (v.s. 90.0%)
- **Market**: -4.1% AR

## Key Parameters

```python
AlphaConfig(
    # GA Parameters
    population_size=100,           # Population size per generation
    max_depth=3,                   # Maximum formula depth
    generations=50,                # Generations per depth level
    warm_start_multiplier=10,      # K for warm start (K × pop_size)
    
    # Evaluation
    holding_period=1,              # Trading holding period (days)
    ic_threshold=0.05,             # Minimum IC for diversity
    similarity_threshold=0.7,      # Alpha similarity threshold
    pca_similarity_threshold=0.9,  # PCA-similarity threshold
    
    # Ensemble
    top_k_alphas=150,              # Top alphas for ensemble
    long_n_stocks=10,              # Number of stocks to long
    transaction_cost=0.003,        # 0.3% transaction cost
    
    # Hyperparameters
    mutation_prob=0.1,             # Mutation probability
    crossover_prob=0.8,            # Crossover probability
    risk_free_rate=0.0,            # Risk-free rate for Sharpe
)
```

## Available Operators

### Basic Operators
- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/` (with safeguard)
- Min: `min(a, b)`
- Max: `max(a, b)`

### Technical Indicators
- Moving Average: `mean(price, window)`
- Standard Deviation: `std(price, window)`
- Time Rank: `tsrank(price, window)`
- Correlation: `corr(x, y, window)`
- Covariance: `cov(x, y, window)`
- Momentum: `momentum(price, window)`
- Volatility: `volatility(returns, window)`

### Raw Factors
- `open` - Opening price
- `close` - Closing price
- `high` - Daily high
- `low` - Daily low
- `volume` - Trading volume
- `vwap` - Volume weighted average price
- `returns` - Daily returns

## Example Discovered Alphas

From paper backtesting:

```
Alpha#1: IC=8.36%
  (std(close, 5) / mean(volume, 20))

Alpha#2: IC=8.30%
  ((close - open) / max(high - low, 0.0001))

Alpha#3: IC=7.84%
  (tsrank(volume, 10) * momentum(close, 20))
```

## Backtesting & Evaluation

### IC (Information Coefficient)
Measures rank correlation between alpha and returns:
```python
IC = mean(corr(alpha_values, stock_returns))
```

### Out-of-Sample Testing
Train on 2010-01-01 to 2017-08-31, test on 2017-09-01 to 2019-07-31

### Stratified Analysis
Divide stocks into 10 deciles by alpha, compare returns across deciles

## Contributing

Contributions welcome! Areas of interest:
- Additional technical indicators
- Alternative diversity mechanisms
- Performance optimizations
- Backtesting enhancements

## References

[1] Zhang, T., Li, Y., Jin, Y., & Li, J. (2020). AutoAlpha: an Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors in Quantitative Investment. *arXiv preprint arXiv:2002.08245*.

[2] Kakushadze, Z. (2016). 101 formulaic alphas. *Wilmott*, 2016(84), 72-81.

[3] Grinold, R. C., & Kahn, R. N. (2000). Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk.

## License

MIT License - see LICENSE file for details

## Citation

If you use AutoAlpha in your research, please cite:

```bibtex
@article{zhang2020autoalpha,
  title={AutoAlpha: an Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors in Quantitative Investment},
  author={Zhang, Tianping and Li, Yuanqi and Jin, Yifei and Li, Jian},
  journal={arXiv preprint arXiv:2002.08245},
  year={2020}
}
```

## Disclaimer

This is an educational implementation for research purposes. Not financial advice. Backtesting results do not guarantee future performance. Always validate strategies with proper risk management.
