"""Tree-based representation of formulaic alphas."""

import copy
import numpy as np
from typing import List, Optional, Tuple, Union
import random


class Node:
    """Represents a node in the alpha expression tree."""
    
    def __init__(self, node_type: str, value=None, children: List['Node'] = None):
        """
        Args:
            node_type: 'operator', 'indicator', or 'factor'
            value: operator/indicator/factor name
            children: child nodes
        """
        self.node_type = node_type  # 'operator', 'indicator', 'factor'
        self.value = value
        self.children = children or []
    
    def __repr__(self) -> str:
        if self.node_type == 'factor':
            return str(self.value)
        elif self.node_type == 'indicator':
            if self.children:
                child_str = ', '.join(repr(c) for c in self.children)
                return f"{self.value}({child_str})"
            return self.value
        else:  # operator
            if len(self.children) == 2:
                return f"({repr(self.children[0])} {self.value} {repr(self.children[1])})"
            elif len(self.children) == 1:
                return f"{self.value}({repr(self.children[0])})"
            return str(self.value)
    
    def depth(self) -> int:
        """Calculate tree depth."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def copy(self) -> 'Node':
        """Deep copy of node and its children."""
        new_children = [child.copy() for child in self.children]
        return Node(self.node_type, self.value, new_children)
    
    def get_all_nodes(self) -> List['Node']:
        """Get all nodes in subtree (DFS)."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes


class AlphaTree:
    """Represents a complete alpha factor expression as a tree."""
    
    # Basic operators
    OPERATORS = ['+', '-', '*', '/', 'max', 'min']
    
    # Technical indicators (with window parameter)
    INDICATORS = {
        'mean': 1,      # (values, window)
        'std': 1,       # (values, window)
        'tsrank': 1,    # (values, window)
        'momentum': 1,  # (values, window)
        'corr': 2,      # (x, y, window)
        'cov': 2,       # (x, y, window)
    }
    
    # Raw factors (always available)
    FACTORS = ['open', 'close', 'high', 'low', 'volume', 'vwap', 'returns']
    
    def __init__(self, root: Optional[Node] = None):
        self.root = root
    
    @property
    def depth(self) -> int:
        """Get tree depth."""
        return self.root.depth() if self.root else 0
    
    @property
    def expression(self) -> str:
        """Get expression string representation."""
        return repr(self.root) if self.root else ''
    
    def copy(self) -> 'AlphaTree':
        """Deep copy of alpha tree."""
        new_root = self.root.copy() if self.root else None
        return AlphaTree(new_root)
    
    @staticmethod
    def random_node(max_depth: int, current_depth: int = 0, must_operator: bool = False) -> Node:
        """Generate random tree node.
        
        Args:
            max_depth: Maximum tree depth
            current_depth: Current depth in tree
            must_operator: Force operator node (for internal nodes)
        
        Returns:
            Random node
        """
        if current_depth >= max_depth or (current_depth > 0 and not must_operator and random.random() < 0.3):
            # Generate leaf node (factor or indicator)
            if random.random() < 0.7:
                # Factor node
                return Node('factor', random.choice(AlphaTree.FACTORS))
            else:
                # Indicator node
                indicator = random.choice(list(AlphaTree.INDICATORS.keys()))
                n_children = AlphaTree.INDICATORS[indicator]
                children = [AlphaTree.random_node(max_depth, current_depth + 1) 
                           for _ in range(n_children)]
                return Node('indicator', indicator, children)
        else:
            # Operator node
            operator = random.choice(AlphaTree.OPERATORS)
            n_children = 2 if operator in ['+', '-', '*', '/', 'max', 'min'] else 1
            children = [AlphaTree.random_node(max_depth, current_depth + 1, must_operator=True)
                       for _ in range(n_children)]
            return Node('operator', operator, children)
    
    @staticmethod
    def random_tree(depth: int) -> 'AlphaTree':
        """Generate random alpha tree with specified depth."""
        root = AlphaTree.random_node(depth)
        return AlphaTree(root)
    
    @staticmethod
    def crossover(alpha1: 'AlphaTree', alpha2: 'AlphaTree', target_depth: int) -> Tuple['AlphaTree', 'AlphaTree']:
        """Crossover two alpha trees at same depth.
        
        Args:
            alpha1: First parent tree
            alpha2: Second parent tree
            target_depth: Depth level to perform crossover
        
        Returns:
            Two offspring trees
        """
        # Deep copy parents
        child1 = alpha1.copy()
        child2 = alpha2.copy()
        
        # Get nodes at target depth
        nodes1 = AlphaTree._get_nodes_at_depth(child1.root, target_depth)
        nodes2 = AlphaTree._get_nodes_at_depth(child2.root, target_depth)
        
        if not nodes1 or not nodes2:
            return child1, child2
        
        # Randomly select nodes to swap
        node1 = random.choice(nodes1)
        node2 = random.choice(nodes2)
        
        # Swap subtrees (copy to avoid shared references)
        temp = node1.children.copy()
        node1.children = [c.copy() for c in node2.children]
        node2.children = temp
        
        return child1, child2
    
    @staticmethod
    def mutation(alpha: 'AlphaTree', mutation_depth: int) -> 'AlphaTree':
        """Randomly mutate alpha tree.
        
        Args:
            alpha: Tree to mutate
            mutation_depth: Maximum depth of mutated subtree
        
        Returns:
            Mutated tree
        """
        mutant = alpha.copy()
        
        # Get all nodes
        all_nodes = mutant.root.get_all_nodes()
        if not all_nodes:
            return mutant
        
        # Select random node and replace its subtree
        node = random.choice(all_nodes)
        if node.children:
            # Replace random child
            idx = random.randint(0, len(node.children) - 1)
            node.children[idx] = AlphaTree.random_node(mutation_depth)
        
        return mutant
    
    @staticmethod
    def _get_nodes_at_depth(node: Optional[Node], target_depth: int, current_depth: int = 1) -> List[Node]:
        """Get all nodes at specific depth."""
        if not node:
            return []
        
        if current_depth == target_depth:
            return [node]
        
        if current_depth > target_depth:
            return []
        
        result = []
        for child in node.children:
            result.extend(AlphaTree._get_nodes_at_depth(child, target_depth, current_depth + 1))
        return result
