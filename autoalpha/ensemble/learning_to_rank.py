"""Learning-to-rank ensemble model for portfolio generation.

Uses LightGBM and XGBoost to rank stocks based on alpha factors,
then selects top N stocks for long-only portfolio.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class LearningToRank:
    """Ensemble learning-to-rank model for portfolio generation."""
    
    def __init__(self,
                 use_lightgbm: bool = True,
                 use_xgboost: bool = True,
                 lightgbm_params: Optional[dict] = None,
                 xgboost_params: Optional[dict] = None):
        """
        Args:
            use_lightgbm: Use LightGBM in ensemble
            use_xgboost: Use XGBoost in ensemble
            lightgbm_params: LightGBM parameters
            xgboost_params: XGBoost parameters
        """
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        
        self.lightgbm_model = None
        self.xgboost_model = None
        
        self.lightgbm_params = lightgbm_params or {
            'objective': 'rank:ndcg',
            'metric': 'ndcg',
            'num_leaves': 31,
            'learning_rate': 0.05,
        }
        
        self.xgboost_params = xgboost_params or {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg',
            'max_depth': 6,
            'learning_rate': 0.05,
        }
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             groups_train: Optional[List[int]] = None) -> None:
        """Train ensemble models.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,) - stock returns or rankings
            groups_train: Group sizes for ranking tasks
        """
        if self.use_lightgbm:
            try:
                train_data = lgb.Dataset(
                    X_train, label=y_train,
                    group=groups_train
                )
                self.lightgbm_model = lgb.train(
                    self.lightgbm_params,
                    train_data,
                    num_boost_round=100
                )
            except Exception as e:
                print(f"LightGBM training error: {e}")
        
        if self.use_xgboost:
            try:
                train_data = xgb.DMatrix(
                    X_train, label=y_train,
                    group=groups_train
                )
                self.xgboost_model = xgb.train(
                    self.xgboost_params,
                    train_data,
                    num_boost_round=100
                )
            except Exception as e:
                print(f"XGBoost training error: {e}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict rankings for test data.
        
        Args:
            X_test: Test features (n_samples, n_features)
        
        Returns:
            Predicted scores (n_samples,) - higher is better
        """
        predictions = []
        
        if self.use_lightgbm and self.lightgbm_model:
            lgb_pred = self.lightgbm_model.predict(X_test)
            predictions.append(lgb_pred)
        
        if self.use_xgboost and self.xgboost_model:
            xgb_data = xgb.DMatrix(X_test)
            xgb_pred = self.xgboost_model.predict(xgb_data)
            predictions.append(xgb_pred)
        
        if not predictions:
            return np.zeros(len(X_test))
        
        # Ensemble: average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def rank_stocks(self,
                   X_test: np.ndarray,
                   n_stocks: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Rank stocks and select top N for portfolio.
        
        Args:
            X_test: Test features (n_stocks, n_features)
            n_stocks: Number of stocks to select
        
        Returns:
            (indices of top stocks, predicted scores)
        """
        scores = self.predict(X_test)
        
        # Get indices of top n_stocks
        top_indices = np.argsort(scores)[-n_stocks:][::-1]
        
        return top_indices, scores[top_indices]
