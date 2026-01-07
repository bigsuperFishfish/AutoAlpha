"""Unit tests for alpha generation and evaluation."""

import unittest
import numpy as np
from autoalpha.core.alpha_tree import AlphaTree, Node
from autoalpha.core.operators import Operators
from autoalpha.core.evaluator import Evaluator
from autoalpha.data.factor_calculator import FactorCalculator


class TestAlphaTree(unittest.TestCase):
    """Test AlphaTree structure and operations."""
    
    def test_node_creation(self):
        """Test node creation."""
        node = Node('factor', 'close')
        self.assertEqual(node.value, 'close')
        self.assertEqual(node.depth(), 1)
    
    def test_tree_depth(self):
        """Test tree depth calculation."""
        # Create simple tree: (close + open) / high
        close_node = Node('factor', 'close')
        open_node = Node('factor', 'open')
        high_node = Node('factor', 'high')
        
        add_node = Node('operator', '+', [close_node, open_node])
        div_node = Node('operator', '/', [add_node, high_node])
        
        tree = AlphaTree(div_node)
        self.assertEqual(tree.depth, 3)
    
    def test_random_tree_generation(self):
        """Test random tree generation."""
        tree = AlphaTree.random_tree(depth=2)
        self.assertIsNotNone(tree.root)
        self.assertLessEqual(tree.depth, 2)


class TestOperators(unittest.TestCase):
    """Test mathematical operators."""
    
    def setUp(self):
        """Set up test data."""
        self.a = np.array([[1, 2], [3, 4]], dtype=float)
        self.b = np.array([[2, 2], [2, 2]], dtype=float)
    
    def test_add(self):
        """Test addition operator."""
        result = Operators.add(self.a, self.b)
        expected = np.array([[3, 4], [5, 6]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_divide_by_zero_safety(self):
        """Test division by zero handling."""
        b_zero = np.array([[0, 0], [0, 0]], dtype=float)
        result = Operators.divide(self.a, b_zero)
        # Should not raise error, should handle gracefully
        self.assertEqual(result.shape, self.a.shape)
        self.assertFalse(np.all(np.isnan(result)))
    
    def test_moving_average(self):
        """Test moving average calculation."""
        x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
        result = Operators.mean(x, window=2)
        # First window: [1,2] and [2,3], means are 1.5 and 2.5
        self.assertEqual(result.shape, x.shape)


class TestEvaluator(unittest.TestCase):
    """Test fitness evaluation functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.T, self.N = 100, 20  # 100 days, 20 stocks
        self.alpha = np.random.randn(self.T, self.N)
        self.returns = self.alpha + np.random.randn(self.T, self.N) * 0.1
    
    def test_information_coefficient(self):
        """Test IC calculation."""
        ic = Evaluator.information_coefficient(self.alpha, self.returns)
        # IC should be between -1 and 1
        self.assertGreaterEqual(ic, -1.0)
        self.assertLessEqual(ic, 1.0)
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = np.random.randn(100) * 0.01 + 0.0005
        sharpe = Evaluator.sharpe_ratio(returns)
        # Sharpe should be a reasonable number
        self.assertIsInstance(sharpe, (float, np.floating))
    
    def test_factor_returns(self):
        """Test factor returns calculation."""
        ic, sharpe, win_rate = Evaluator.factor_returns(self.alpha, self.returns)
        # Check return types and ranges
        self.assertIsInstance(ic, (float, np.floating))
        self.assertIsInstance(sharpe, (float, np.floating))
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)


class TestFactorCalculator(unittest.TestCase):
    """Test factor expression calculation."""
    
    def setUp(self):
        """Set up test data."""
        self.data = {
            'close': np.random.randn(50, 10) + 100,
            'open': np.random.randn(50, 10) + 100,
            'high': np.random.randn(50, 10) + 101,
            'low': np.random.randn(50, 10) + 99,
        }
    
    def test_factor_calculation(self):
        """Test basic factor calculation."""
        calculator = FactorCalculator(self.data)
        
        # Create simple tree: close
        close_node = Node('factor', 'close')
        tree = AlphaTree(close_node)
        
        result = calculator.calculate(tree)
        np.testing.assert_array_almost_equal(result, self.data['close'])
    
    def test_operator_calculation(self):
        """Test operator calculation."""
        calculator = FactorCalculator(self.data)
        
        # Create tree: (close - open) / high
        close_node = Node('factor', 'close')
        open_node = Node('factor', 'open')
        high_node = Node('factor', 'high')
        
        sub_node = Node('operator', '-', [close_node, open_node])
        div_node = Node('operator', '/', [sub_node, high_node])
        
        tree = AlphaTree(div_node)
        result = calculator.calculate(tree)
        
        # Check shape
        self.assertEqual(result.shape, self.data['close'].shape)
        # Check values are reasonable (not all NaN)
        self.assertGreater(np.sum(~np.isnan(result)), 0)


if __name__ == '__main__':
    unittest.main()
