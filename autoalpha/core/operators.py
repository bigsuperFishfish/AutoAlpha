"""Mathematical operators and technical indicators for alpha expressions."""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy.stats import rankdata


class Operators:
    """Collection of operators and indicators for alpha calculation."""
    
    # Epsilon for division safety
    EPS = 1e-8
    
    @staticmethod
    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise addition."""
        return a + b
    
    @staticmethod
    def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise subtraction."""
        return a - b
    
    @staticmethod
    def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiplication."""
        return a * b
    
    @staticmethod
    def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise division with safeguard."""
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            result = a / np.where(np.abs(b) < Operators.EPS, Operators.EPS, b)
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result
    
    @staticmethod
    def max_op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise maximum."""
        return np.maximum(a, b)
    
    @staticmethod
    def min_op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise minimum."""
        return np.minimum(a, b)
    
    @staticmethod
    def mean(x: np.ndarray, window: int = 20) -> np.ndarray:
        """Moving average.
        
        Args:
            x: Input time series (T, N) where T=time, N=stocks
            window: Window size
        
        Returns:
            Moving average (T, N)
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        result = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            start = max(0, i - window + 1)
            result[i] = np.mean(x[start:i+1], axis=0)
        return result
    
    @staticmethod
    def std(x: np.ndarray, window: int = 20) -> np.ndarray:
        """Moving standard deviation.
        
        Args:
            x: Input time series (T, N)
            window: Window size
        
        Returns:
            Moving std (T, N)
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        result = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            start = max(0, i - window + 1)
            result[i] = np.std(x[start:i+1], axis=0)
        return result
    
    @staticmethod
    def tsrank(x: np.ndarray, window: int = 20) -> np.ndarray:
        """Time-series rank within rolling window.
        
        Args:
            x: Input time series (T, N)
            window: Window size
        
        Returns:
            Rank (T, N)
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        T, N = x.shape
        result = np.zeros_like(x, dtype=float)
        
        for i in range(T):
            start = max(0, i - window + 1)
            window_data = x[start:i+1]  # (w, N)
            
            for j in range(N):
                col = window_data[:, j]
                # Rank: 1 to len(col)
                rank = rankdata(col) / len(col)
                result[i, j] = rank[-1]
        
        return result
    
    @staticmethod
    def corr(x: np.ndarray, y: np.ndarray, window: int = 20) -> np.ndarray:
        """Moving correlation.
        
        Args:
            x: Input series (T, N)
            y: Input series (T, N)
            window: Window size
        
        Returns:
            Correlation (T, N)
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        T, N = x.shape
        result = np.zeros((T, N), dtype=float)
        
        for i in range(T):
            start = max(0, i - window + 1)
            x_window = x[start:i+1]  # (w, N)
            y_window = y[start:i+1]  # (w, N)
            
            # Calculate correlation per stock
            for j in range(N):
                x_col = x_window[:, j]
                y_col = y_window[:, j]
                
                if len(x_col) > 1 and np.std(x_col) > Operators.EPS and np.std(y_col) > Operators.EPS:
                    result[i, j] = np.corrcoef(x_col, y_col)[0, 1]
        
        return result
    
    @staticmethod
    def cov(x: np.ndarray, y: np.ndarray, window: int = 20) -> np.ndarray:
        """Moving covariance.
        
        Args:
            x: Input series (T, N)
            y: Input series (T, N)
            window: Window size
        
        Returns:
            Covariance (T, N)
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        T, N = x.shape
        result = np.zeros((T, N), dtype=float)
        
        for i in range(T):
            start = max(0, i - window + 1)
            x_window = x[start:i+1]
            y_window = y[start:i+1]
            
            for j in range(N):
                result[i, j] = np.cov(x_window[:, j], y_window[:, j])[0, 1]
        
        return result
    
    @staticmethod
    def momentum(x: np.ndarray, window: int = 20) -> np.ndarray:
        """Rate of change / momentum.
        
        Args:
            x: Input series (T, N)
            window: Window size
        
        Returns:
            Momentum (T, N)
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        result = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            if i >= window:
                result[i] = (x[i] - x[i - window]) / np.where(np.abs(x[i - window]) < Operators.EPS, 
                                                               Operators.EPS, np.abs(x[i - window]))
        
        return result
