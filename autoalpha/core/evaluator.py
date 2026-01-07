"""Fitness evaluation functions: IC (Information Coefficient) and Sharpe Ratio."""

import numpy as np
from typing import Tuple
from scipy.stats import spearmanr, pearsonr


class Evaluator:
    """Evaluates alpha factor quality using IC and Sharpe ratio."""
    
    @staticmethod
    def information_coefficient(alpha: np.ndarray, returns: np.ndarray, method: str = 'pearson') -> float:
        """Calculate Information Coefficient (IC).
        
        IC = mean of daily correlations between alpha values and future returns.
        Higher IC indicates stronger predictive power.
        
        Args:
            alpha: Alpha values (T, N) where T=time, N=stocks
            returns: Stock returns (T, N), same shape as alpha
            method: 'pearson' or 'spearman'
        
        Returns:
            IC value (typically between -1 and 1, but can be expressed as percentage)
        """
        if alpha.shape != returns.shape:
            raise ValueError(f"Alpha shape {alpha.shape} != returns shape {returns.shape}")
        
        T, N = alpha.shape
        daily_corrs = []
        
        for t in range(T):
            alpha_t = alpha[t]  # (N,)
            returns_t = returns[t]  # (N,)
            
            # Remove NaN values
            mask = ~(np.isnan(alpha_t) | np.isnan(returns_t))
            if np.sum(mask) < 2:
                continue
            
            alpha_clean = alpha_t[mask]
            returns_clean = returns_t[mask]
            
            if method == 'pearson':
                corr = np.corrcoef(alpha_clean, returns_clean)[0, 1]
            else:  # spearman
                corr, _ = spearmanr(alpha_clean, returns_clean)
            
            if not np.isnan(corr):
                daily_corrs.append(corr)
        
        if not daily_corrs:
            return 0.0
        
        return np.mean(daily_corrs)
    
    @staticmethod
    def sharpe_ratio(returns_strategy: np.ndarray, risk_free_rate: float = 0.0, 
                     trading_days_per_year: int = 252) -> float:
        """Calculate Sharpe Ratio.
        
        Sharpe = (mean_return - risk_free_rate) / std_return
        Annualized based on trading_days_per_year.
        
        Args:
            returns_strategy: Daily returns (T,) or (T, N) where T=time
            risk_free_rate: Annual risk-free rate (default 0)
            trading_days_per_year: Days per year (default 252)
        
        Returns:
            Annualized Sharpe ratio
        """
        if returns_strategy.ndim > 1:
            # If portfolio returns (multiple stocks), take mean or sum
            returns_strategy = np.mean(returns_strategy, axis=1)
        
        # Remove NaN
        returns_clean = returns_strategy[~np.isnan(returns_strategy)]
        
        if len(returns_clean) < 2:
            return 0.0
        
        daily_mean_return = np.mean(returns_clean)
        daily_std_return = np.std(returns_clean)
        
        # Annualize
        annual_mean_return = daily_mean_return * trading_days_per_year
        annual_std_return = daily_std_return * np.sqrt(trading_days_per_year)
        
        if annual_std_return < 1e-8:
            return 0.0
        
        sharpe = (annual_mean_return - risk_free_rate) / annual_std_return
        return sharpe
    
    @staticmethod
    def factor_returns(alpha: np.ndarray, returns: np.ndarray) -> Tuple[float, float, float]:
        """Calculate factor statistics: IC, Sharpe, and win rate.
        
        Args:
            alpha: Alpha values (T, N)
            returns: Stock returns (T, N)
        
        Returns:
            (ic, sharpe, win_rate)
        """
        ic = Evaluator.information_coefficient(alpha, returns)
        
        # Calculate factor returns: correlation-weighted
        T, N = alpha.shape
        factor_returns_ts = []
        
        for t in range(T):
            alpha_t = alpha[t]
            returns_t = returns[t]
            mask = ~(np.isnan(alpha_t) | np.isnan(returns_t))
            
            if np.sum(mask) > 1:
                # Normalize alpha to get weights
                alpha_norm = alpha_t[mask]
                if np.std(alpha_norm) > 1e-8:
                    weights = (alpha_norm - np.mean(alpha_norm)) / np.std(alpha_norm)
                    weighted_return = np.sum(weights * returns_t[mask]) / np.sum(np.abs(weights))
                    factor_returns_ts.append(weighted_return)
        
        sharpe = Evaluator.sharpe_ratio(np.array(factor_returns_ts))
        
        # Win rate: percentage of days with positive returns
        win_rate = np.mean(np.array(factor_returns_ts) > 0) if factor_returns_ts else 0.0
        
        return ic, sharpe, win_rate
