from typing import Dict

import numpy as np
import torch


class POSMDPBuffer:
    """
    Trajectory Buffer optimized for asynchronous Multi-Agent transitions.

    Stores trajectories including the time-interval (Delta T) between
    events, allowing for continuous-time reward discounting and Neural
    ODE hidden state evolution during optimization.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_shape: tuple,
        hidden_dim: int,
        global_state_dim: int,
    ):
        self.capacity = capacity
        self.num_agents = num_agents

        # Continuous-Time Observation Buffers
        self.obs = torch.zeros((capacity, num_agents, *obs_shape))
        self.h_states = torch.zeros((capacity, num_agents, hidden_dim))
        self.c_states = torch.zeros(
            (capacity, num_agents, hidden_dim)
        )  # LSTM cell states
        self.delta_ts = torch.zeros((capacity, num_agents, 1))
        self.siem_embeddings = torch.zeros((capacity, num_agents, 128))

        # Action & Reward Buffers
        self.actions = torch.zeros((capacity, num_agents, 2), dtype=torch.long)
        self.rewards = torch.zeros((capacity, num_agents, 1))
        self.masks = torch.zeros(
            (capacity, num_agents, 82)
        )  # Action masking (32 types + 50 targets)
        self.log_probs = torch.zeros(
            (capacity, num_agents, 1)
        )  # Policy log-probabilities

        # Centralized Critic Buffers (Ground Truth)
        self.global_states = torch.zeros((capacity, global_state_dim))
        self.advantages = torch.zeros((capacity, num_agents, 1))
        self.returns = torch.zeros((capacity, num_agents, 1))

        # Transition Continuity (for TD-Learning Baselines)
        self.next_obs = torch.zeros((capacity, num_agents, *obs_shape))
        self.next_global_states = torch.zeros((capacity, global_state_dim))

        # Flags
        self.dones = torch.zeros((capacity, num_agents, 1))

        self.ptr = 0
        self.is_full = False

    def insert(
        self,
        obs: torch.Tensor,
        h_state: torch.Tensor,
        c_state: torch.Tensor,
        dt: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        mask: torch.Tensor,
        global_state: torch.Tensor,
        next_obs: torch.Tensor,
        next_global_state: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        siem_emb: torch.Tensor = None,
    ):
        """
        Inserts a multi-agent transition into the buffer.
        """
        self.obs[self.ptr] = obs
        self.h_states[self.ptr] = h_state
        self.c_states[self.ptr] = c_state
        self.delta_ts[self.ptr] = dt
        if siem_emb is not None:
            self.siem_embeddings[self.ptr] = siem_emb
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.masks[self.ptr] = mask
        self.log_probs[self.ptr] = log_prob
        self.global_states[self.ptr] = global_state
        self.next_obs[self.ptr] = next_obs
        self.next_global_states[self.ptr] = next_global_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.is_full = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Samples a random batch of transitions for the MAPPO update.

        Returns:
            Dict: Dictionary of tensors optimized for PyTorch batch processing.
        """
        upper = self.capacity if self.is_full else self.ptr
        # Guard: Sample size cannot exceed population
        batch_size = min(batch_size, upper)
        if batch_size == 0:
            return {}

        idxs = np.random.choice(upper, batch_size, replace=False)

        return {
            'obs': self.obs[idxs],
            'h_prev': self.h_states[idxs],
            'c_prev': self.c_states[idxs],
            'dt': self.delta_ts[idxs],
            'siem_emb': self.siem_embeddings[idxs],
            'actions': self.actions[idxs],
            'rewards': self.rewards[idxs],
            'masks': self.masks[idxs],
            'log_probs': self.log_probs[idxs],
            'global_state': self.global_states[idxs],
            'next_obs': self.next_obs[idxs],
            'next_global_state': self.next_global_states[idxs],
            'done': self.dones[idxs],
            'advantages': self.advantages[idxs],
            'returns': self.returns[idxs],
        }

    def clear(self):
        """Clears the buffer for the next optimization epoch."""
        self.ptr = 0
        self.is_full = False
