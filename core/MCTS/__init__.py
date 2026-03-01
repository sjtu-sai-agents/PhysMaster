"""
MCTS (Monte Carlo Tree Search) module for multi-agent physics reasoning.

This module implements the complete MCTS algorithm with:
- Selection (UCB1)
- Expansion (parallel node generation)
- Simulation (node execution)
- Backpropagation (reward propagation)
"""

from .mcts import MCTSNode
from .mcts import MCTSTree

try:
    from .supervisor import SupervisorOrchestrator
except Exception:  # pragma: no cover - optional heavy deps may be missing in lightweight envs
    SupervisorOrchestrator = None

try:
    from .theoretician import run_theo_node
except Exception:  # pragma: no cover
    run_theo_node = None

__all__ = ["MCTSNode", "MCTSTree", "SupervisorOrchestrator", "run_theo_node"]
