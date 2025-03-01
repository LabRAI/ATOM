"""
attack_manager.py

This module manages different model extraction attack strategies, allowing flexible selection 
of attack methods such as IGP, AGE, and Grain based on the specified attack type.

Classes:
- ModelExtractionAttack:
  - Provides a unified interface to execute different model extraction attacks.
  - Supports IGP (Influence Graph Propagation), AGE (Active Graph Extraction), and Grain attacks.
  - Handles attack initialization, query tracking, and result synchronization.

Methods:
- __init__(attack_type, target_model, extracted_model, subgraph_access_fn, query_budget, NCL, num_sample_nodes=None, gamma=None, alpha=None):
  - Initializes the attack based on the specified type and required parameters.

- _sync_queries():
  - Synchronizes query information between the selected attack implementation and the manager.

- attack(*args, **kwargs):
  - Executes the selected attack method and updates the query history.

- generate_queries(*args, **kwargs):
  - Delegates query generation to the underlying attack implementation.

- print_query_info():
  - Displays the collected query information for analysis.

Dependencies:
- IGPAttack (influence-based attack)
- AGEAttack (active learning-based attack)
- GrainAttack (graph-based attack)
"""


from .igp_attack import IGPAttack
from .age_attack import AGEAttack
from .grain_attack import GrainAttack

class ModelExtractionAttack:
    def __init__(
        self,
        attack_type,
        target_model,
        extracted_model,
        subgraph_access_fn,
        query_budget,
        NCL,
        num_sample_nodes=None,
        gamma=None,
        alpha=None
    ):
        self.attack_type = attack_type.lower()
        self.target_model = target_model
        self.extracted_model = extracted_model
        self.subgraph_access_fn = subgraph_access_fn
        self.query_budget = query_budget
        self.num_sample_nodes = num_sample_nodes
        self.NCL = NCL

        self.queries_local = []
        self.queries_global = []

        if self.attack_type == "igp":
            if alpha is None:
                raise ValueError("[IGP] Requires alpha parameter.")
            self._attack_impl = IGPAttack(
                target_model,
                extracted_model,
                subgraph_access_fn,
                query_budget,
                alpha
            )
        elif self.attack_type == "age":
            self._attack_impl = AGEAttack(
                target_model,
                extracted_model,
                subgraph_access_fn,
                query_budget,
                NCL
            )
        elif self.attack_type == "grain":
            if gamma is None:
                raise ValueError("[Grain] Requires gamma parameter.")
            if num_sample_nodes is None:
                raise ValueError("[Grain] Requires the num_sample_nodes parameter.")
            self._attack_impl = GrainAttack(
                target_model,
                extracted_model,
                subgraph_access_fn,
                query_budget,
                num_sample_nodes,
                gamma
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}.")

    def _sync_queries(self):
        if hasattr(self._attack_impl, "queries_local"):
            self.queries_local = self._attack_impl.queries_local
        if hasattr(self._attack_impl, "queries_global"):
            self.queries_global = self._attack_impl.queries_global

    def attack(self, *args, **kwargs):
        result = self._attack_impl.attack(*args, **kwargs)
        self._sync_queries()
        return result

    def generate_queries(self, *args, **kwargs):
        return self._attack_impl.generate_queries(*args, **kwargs)

    def print_query_info(self):
        print("=== Query information ===")
        print(f"[Local] {self.queries_local}")
        print(f"[Global] {self.queries_global}")
