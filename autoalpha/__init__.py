"""AutoAlpha - Hierarchical Evolutionary Algorithm for Alpha Factor Mining"""

__version__ = '0.1.0'
__author__ = 'AutoAlpha Contributors'

from autoalpha.core import AlphaTree, GeneticAlgorithm, Evaluator
from autoalpha.search import HierarchicalSearch, PCA_QD
from autoalpha.ensemble import LearningToRank

__all__ = [
    'AlphaTree',
    'GeneticAlgorithm',
    'Evaluator',
    'HierarchicalSearch',
    'PCA_QD',
    'LearningToRank',
]
