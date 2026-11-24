"""Self-rewarding data factory for TOLSA Task 2 ground-truth augmentation.

Adapts the Self-Rewarding Language Models idea (Yuan et al., ICML 2024) to
TOLSA: one LLM acts as Writer (generates new annotated prediction examples)
and as Judge (scores them with an additive rubric against the TOLSA schema).
Only judge-approved examples are kept as candidate ground truth.
"""

__version__ = "0.1.0"
