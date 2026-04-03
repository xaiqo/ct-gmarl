from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class BaseAgent(nn.Module, ABC):
    """
    Abstract Base Class for Multi-Agent Reinforcement Learning Policies.

    NOTE:
    - CT-GMARL (Continuous-Time Graph MARL) is the POLICY ARCHITECTURE, responsible for
      spatial reasoning (GAT) and asynchronous temporal integration (Neural ODEs).
    - MAPPO (Multi-Agent PPO) is strictly the MATHEMATICAL OPTIMIZER used by the
      Trainer to perform gradient updates on the CT-GMARL weights.
    """

    def __init__(self, config: Dict[str, Any]):
        super(BaseAgent, self).__init__()
        self.config = config

    @abstractmethod
    def init_hidden(self, batch_size: int, device: torch.device) -> Any:
        """
        Initializes the algorithm-specific hidden state (e.g., Tensor or Tuple).
        """
        pass

    @abstractmethod
    def select_action(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
        dt: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Samples an action from the policy.

        Args:
            obs: Local observation tensor [Batch, ...].
            h_prev: Previous hidden state [Batch, Hidden].
            dt: Continuous-time jump [Batch, 1].
            mask: Binary action mask [Batch, ActionDim].

        Returns:
            action: Selected action [Batch, Dimensions].
            log_prob: Action log-probability [Batch, 1].
            h_new: Final hidden state [Batch, Hidden].
            extra: Dict of diagnostic metrics (e.g., ODE NFE).
        """
        pass

    @abstractmethod
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
        dt: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Evaluates actions for batch gradient updates.

        Returns:
            new_log_probs: [Batch, 1]
            entropy: [Batch, 1]
            extra: Dict of diagnostics.
        """
        pass

    @abstractmethod
    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Centralized advantage/value estimation.
        """
        pass
