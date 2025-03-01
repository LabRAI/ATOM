"""
base.py

This module defines the abstract base class `BaseAttack` for implementing different model 
extraction attacks. It provides a standardized interface for attack strategies, ensuring 
that all attack implementations include query generation and execution methods.

Classes:
- BaseAttack (ABC):
  - Abstract base class for defining attack strategies.
  - Enforces implementation of `generate_queries` and `attack` methods in derived classes.

Methods:
- generate_queries(features, adj_matrix, normcen):
  - Abstract method to generate query sequences based on features, adjacency matrix, and centrality.

- attack(data, optimizer, criterion, epochs=10):
  - Abstract method for executing the attack process, training an extracted model iteratively.

Usage:
- This class should be subclassed to implement specific attack strategies such as IGP, AGE, or Grain.
"""


from abc import ABC, abstractmethod

class BaseAttack(ABC):
    @abstractmethod
    def generate_queries(self, features, adj_matrix, normcen):
        pass

    @abstractmethod
    def attack(self, data, optimizer, criterion, epochs=10):
        pass
