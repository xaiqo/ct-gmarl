import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional

from src.models.base import BaseAgent


class QMixer(nn.Module):
    """
    Monotonic Mixing Network for QMIX.
    
    Transforms individual agent Q-values into a joint Q_tot using a 
    hyper-network that enforces monotonicity through non-negative weights.
    """
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hyper-networks for weights
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, n_agents * embed_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

        # Hyper-networks for biases
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Calculates Q_tot = f(Q1, Q2, ..., Qn, S).
        """
        batch_size = agent_qs.shape[0]
        # w1: [batch_size, n_agents, embed_dim]
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
        
        # hidden: [batch_size, 1, embed_dim]
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        # w2: [batch_size, embed_dim, 1]
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, -1)


class QMIXAgent(BaseAgent):
    """
    QMIX Research Agent Baseline.
    
    Implements value-based multi-agent reinforcement learning with discrete-time 
    mixing. Uses GRU for local agent history.
    """

    def __init__(self, config: Dict[str, Any]):
        super(QMIXAgent, self).__init__(config)
        hidden_dim = config['model']['hidden_dim']
        obs_dim = config.get('obs_dim', 256)
        global_in_dim = config.get('global_state_dim', 512)
        n_agents = config.get('n_agents', 4)

        # Local Q-Network (Discrete temporal)
        self.gru = nn.GRUCell(obs_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 62) # 12 types + 50 IPs
        
        # Centralized Mixer
        self.mixer = QMixer(n_agents=n_agents, state_dim=global_in_dim)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config['model']['hidden_dim'], device=device)

    def select_action(
        self, 
        obs: torch.Tensor, 
        h_prev: torch.Tensor, 
        dt: torch.Tensor, 
        mask: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        h_new = self.gru(obs, h_prev)
        q_values = self.q_head(h_new)
        
        # Epsilon-greedy rollout (Heuristic provided by Trainer or fixed here)
        epsilon = kwargs.get('epsilon', 0.05)
        
        # Masking out invalid Q-values
        inf_mask = (1.0 - mask) * -1e9
        masked_qs = q_values + inf_mask
        
        # Greedy selection
        # Note: QMIX usually selects best joint action. Here we select best individual types and IPs.
        # Action space: [12 types, 50 IPs] -> To simplify, we argmax the masked logits.
        
        a_type = torch.argmax(masked_qs[:, :12], dim=-1)
        a_target = torch.argmax(masked_qs[:, 12:], dim=-1)
        
        # Return Q-values as 'log_prob' for uniform API, though MultiTrainer will use them differently
        log_prob = masked_qs.gather(1, a_type.unsqueeze(-1)) # Simplified
        
        return torch.stack([a_type, a_target], dim=-1), log_prob, h_new, {'q_values': q_values}

    def evaluate_actions(self, *args, **kwargs):
        # QMIX doesn't use standard MAPPO evaluate_actions.
        # But we implement a version that returns local Q-values for all actions.
        obs, h_prev, dt, actions, mask = args[0], args[1], args[2], args[3], args[4]
        h_new = self.gru(obs, h_prev)
        q_values = self.q_head(h_new)
        
        return q_values, torch.zeros_like(q_values), {}

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1) # Needs multi-agent Q inputs to be useful

    def get_q_tot(self, agent_qs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        return self.mixer(agent_qs, global_state)
