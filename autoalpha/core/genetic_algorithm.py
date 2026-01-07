"""Core genetic algorithm implementation for alpha factor evolution."""

import numpy as np
import random
from typing import List, Callable, Optional, Tuple
from autoalpha.core.alpha_tree import AlphaTree
from autoalpha.core.evaluator import Evaluator


class Individual:
    """Represents an individual (alpha factor) in the population."""
    
    def __init__(self, alpha: AlphaTree, fitness: float = None):
        self.alpha = alpha
        self.fitness = fitness
        self.ic = None
        self.sharpe = None
    
    def __lt__(self, other):
        """For sorting (by fitness descending)."""
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness > other.fitness
    
    def copy(self):
        ind = Individual(self.alpha.copy())
        ind.fitness = self.fitness
        ind.ic = self.ic
        ind.sharpe = self.sharpe
        return ind


class GeneticAlgorithm:
    """Genetic Algorithm for alpha factor evolution."""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_depth: int = 3,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.1,
                 warm_start_multiplier: int = 10):
        """
        Args:
            population_size: Size of population
            max_depth: Maximum tree depth
            crossover_prob: Crossover probability (0-1)
            mutation_prob: Mutation probability (0-1)
            warm_start_multiplier: K for warm start (K * pop_size candidates)
        """
        self.population_size = population_size
        self.max_depth = max_depth
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.warm_start_multiplier = warm_start_multiplier
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
    
    def initialize_warm_start(self, 
                             population_size: int,
                             fitness_func: Callable[[AlphaTree], float]) -> List[Individual]:
        """Warm start initialization.
        
        Generate K * population_size random candidates, select top 1/K by fitness.
        
        Args:
            population_size: Target population size
            fitness_func: Function to evaluate individual fitness
        
        Returns:
            Initialized population
        """
        n_candidates = population_size * self.warm_start_multiplier
        candidates = []
        
        for _ in range(n_candidates):
            alpha = AlphaTree.random_tree(self.max_depth)
            fitness = fitness_func(alpha)
            individual = Individual(alpha, fitness)
            candidates.append(individual)
        
        # Sort by fitness (descending) and select top 1/K
        candidates.sort(reverse=True)
        population = candidates[:population_size]
        
        return population
    
    def initialize_random(self, 
                         population_size: int,
                         fitness_func: Callable[[AlphaTree], float]) -> List[Individual]:
        """Random initialization.
        
        Args:
            population_size: Population size
            fitness_func: Fitness evaluation function
        
        Returns:
            Random population
        """
        population = []
        for _ in range(population_size):
            alpha = AlphaTree.random_tree(self.max_depth)
            fitness = fitness_func(alpha)
            individual = Individual(alpha, fitness)
            population.append(individual)
        
        return population
    
    def selection_tournament(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        """Tournament selection.
        
        Args:
            population: Current population
            tournament_size: Size of tournament
        
        Returns:
            Selected individual
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness if x.fitness else 0)
    
    def crossover(self, 
                 parent1: Individual, 
                 parent2: Individual,
                 crossover_depth: int) -> Tuple[Individual, Individual]:
        """Crossover two individuals.
        
        Args:
            parent1: First parent
            parent2: Second parent
            crossover_depth: Depth level to perform crossover
        
        Returns:
            Two offspring
        """
        child1_alpha, child2_alpha = AlphaTree.crossover(parent1.alpha, parent2.alpha, crossover_depth)
        child1 = Individual(child1_alpha)
        child2 = Individual(child2_alpha)
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate individual.
        
        Args:
            individual: Individual to mutate
        
        Returns:
            Mutated individual
        """
        mutated_alpha = AlphaTree.mutation(individual.alpha, self.max_depth)
        return Individual(mutated_alpha)
    
    def evolve_generation(self,
                         population: List[Individual],
                         fitness_func: Callable[[AlphaTree], float],
                         replacement_method: str = 'parent_offspring') -> List[Individual]:
        """Execute one generation of evolution.
        
        Args:
            population: Current population
            fitness_func: Fitness evaluation function
            replacement_method: 'best' or 'parent_offspring'
        
        Returns:
            New population
        """
        new_population = population.copy()
        
        # Generate offspring
        offspring = []
        for _ in range(len(population)):
            if random.random() < self.crossover_prob:
                parent1 = self.selection_tournament(population)
                parent2 = self.selection_tournament(population)
                child1, child2 = self.crossover(parent1, parent2, 1)  # Crossover at depth 1
                offspring.extend([child1, child2])
            else:
                parent = self.selection_tournament(population)
                offspring.append(parent.copy())
        
        # Mutation
        for individual in offspring:
            if random.random() < self.mutation_prob:
                individual = self.mutate(individual)
        
        # Evaluate offspring
        for individual in offspring:
            if individual.fitness is None:
                individual.fitness = fitness_func(individual.alpha)
        
        # Replacement
        if replacement_method == 'parent_offspring':
            # Parent-offspring competition
            for offspring_ind in offspring:
                # Replace worst individual if offspring is better
                worst_idx = min(range(len(new_population)), 
                               key=lambda i: new_population[i].fitness if new_population[i].fitness else 0)
                if offspring_ind.fitness > (new_population[worst_idx].fitness or 0):
                    new_population[worst_idx] = offspring_ind
        else:  # 'best'
            new_population.extend(offspring)
            new_population.sort(reverse=True)
            new_population = new_population[:self.population_size]
        
        return new_population
    
    def evolve(self,
              population_size: int,
              n_generations: int,
              fitness_func: Callable[[AlphaTree], float],
              use_warm_start: bool = True) -> List[Individual]:
        """Run genetic algorithm for specified generations.
        
        Args:
            population_size: Population size
            n_generations: Number of generations
            fitness_func: Fitness function
            use_warm_start: Use warm start initialization
        
        Returns:
            Final population
        """
        # Initialize population
        if use_warm_start:
            self.population = self.initialize_warm_start(population_size, fitness_func)
        else:
            self.population = self.initialize_random(population_size, fitness_func)
        
        self.best_individual = max(self.population, key=lambda x: x.fitness or 0)
        
        # Evolution loop
        for gen in range(n_generations):
            self.population = self.evolve_generation(self.population, fitness_func)
            
            current_best = max(self.population, key=lambda x: x.fitness or 0)
            if current_best.fitness > (self.best_individual.fitness or 0):
                self.best_individual = current_best.copy()
        
        return self.population
