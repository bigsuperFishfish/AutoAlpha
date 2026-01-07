"""Factor calculation engine for evaluating alpha expressions on data."""

import numpy as np
from typing import Dict, Optional, Callable
from autoalpha.core.alpha_tree import AlphaTree, Node
from autoalpha.core.operators import Operators


class FactorCalculator:
    """Evaluates alpha tree expressions on market data."""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        """
        Args:
            data: Dictionary of factor data {name: (T, N) array}
                 e.g., {'open': price_array, 'close': price_array, ...}
        """
        self.data = data
        self.cache = {}  # Cache for computed nodes
    
    def calculate(self, alpha: AlphaTree) -> np.ndarray:
        """Evaluate alpha expression on data.
        
        Args:
            alpha: AlphaTree to evaluate
        
        Returns:
            Alpha values (T, N) for each time and stock
        """
        self.cache.clear()
        
        if not alpha.root:
            return np.zeros((len(next(iter(self.data.values()))), 
                           len(next(iter(self.data.values()))[0])))
        
        return self._evaluate_node(alpha.root)
    
    def _evaluate_node(self, node: Node) -> np.ndarray:
        """Recursively evaluate a node.
        
        Args:
            node: Node to evaluate
        
        Returns:
            Evaluated values (T, N)
        """
        # Check cache
        node_id = id(node)
        if node_id in self.cache:
            return self.cache[node_id]
        
        result = None
        
        if node.node_type == 'factor':
            # Leaf node: raw factor
            factor_name = node.value
            if factor_name in self.data:
                result = self.data[factor_name].copy()
            else:
                raise ValueError(f"Factor '{factor_name}' not found in data")
        
        elif node.node_type == 'operator':
            # Operator node
            op = node.value
            
            if op == '+':
                left = self._evaluate_node(node.children[0])
                right = self._evaluate_node(node.children[1])
                result = Operators.add(left, right)
            
            elif op == '-':
                left = self._evaluate_node(node.children[0])
                right = self._evaluate_node(node.children[1])
                result = Operators.subtract(left, right)
            
            elif op == '*':
                left = self._evaluate_node(node.children[0])
                right = self._evaluate_node(node.children[1])
                result = Operators.multiply(left, right)
            
            elif op == '/':
                left = self._evaluate_node(node.children[0])
                right = self._evaluate_node(node.children[1])
                result = Operators.divide(left, right)
            
            elif op == 'max':
                left = self._evaluate_node(node.children[0])
                right = self._evaluate_node(node.children[1])
                result = Operators.max_op(left, right)
            
            elif op == 'min':
                left = self._evaluate_node(node.children[0])
                right = self._evaluate_node(node.children[1])
                result = Operators.min_op(left, right)
        
        elif node.node_type == 'indicator':
            # Indicator node
            indicator = node.value
            
            if indicator == 'mean':
                x = self._evaluate_node(node.children[0])
                window = 20  # Default window
                result = Operators.mean(x, window)
            
            elif indicator == 'std':
                x = self._evaluate_node(node.children[0])
                window = 20
                result = Operators.std(x, window)
            
            elif indicator == 'tsrank':
                x = self._evaluate_node(node.children[0])
                window = 20
                result = Operators.tsrank(x, window)
            
            elif indicator == 'momentum':
                x = self._evaluate_node(node.children[0])
                window = 20
                result = Operators.momentum(x, window)
            
            elif indicator == 'corr':
                x = self._evaluate_node(node.children[0])
                y = self._evaluate_node(node.children[1])
                window = 20
                result = Operators.corr(x, y, window)
            
            elif indicator == 'cov':
                x = self._evaluate_node(node.children[0])
                y = self._evaluate_node(node.children[1])
                window = 20
                result = Operators.cov(x, y, window)
        
        if result is None:
            raise ValueError(f"Cannot evaluate node: {node}")
        
        # Cache result
        self.cache[node_id] = result
        return result
