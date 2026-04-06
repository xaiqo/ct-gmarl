from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.models.base import BaseAgent
from src.models.ct_gmarl.gat_processor import MultiHeadGAT, TopologyMessagePasser
from src.models.ct_gmarl.ode_engine import ODERNNCell


class IndependentActionHeads(nn.Module):
    def __init__(self, hidden_dim: int, n_types: int = 32, n_targets: int = 100):
        super(IndependentActionHeads, self).__init__()
        self.policy_type = nn.Linear(hidden_dim, n_types)
        self.policy_target = nn.Linear(hidden_dim, n_targets)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Full Spectrum: First 32 bits are action types
        mask_type = mask[:, :32]
        mask_target = mask[:, 32:]
        logits_type = self.policy_type(h)
        logits_target = self.policy_target(h)
        inf_mask_type = (1.0 - mask_type) * -1e9
        inf_mask_target = (1.0 - mask_target) * -1e9
        return logits_type + inf_mask_type, logits_target + inf_mask_target


class CentralizedNoiselessCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden_dim: int = 256):
        super(CentralizedNoiselessCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)


class CTGMARLAgent(BaseAgent):
    """
    Research Agent implementing the CT-GMARL architecture.

    Combines GAT spatial reasoning with Neural ODE temporal integration
    to solve asynchronous, continuous-time MARL problems.
    """

    def __init__(self, config: Dict[str, Any]):
        super(CTGMARLAgent, self).__init__(config)
        hidden_dim = config['model']['hidden_dim']
        node_in_dim = config.get('node_in_dim', 256)
        global_in_dim = config.get('global_state_dim', 512)
        n_heads = config.get('n_heads', 4)

        self.use_gat = config.get('model', {}).get('use_gat', True)
        self.use_ode = config.get('model', {}).get('use_ode', True)

        if self.use_gat:
            self.gat = MultiHeadGAT(
                in_features=node_in_dim, n_hidden=hidden_dim, n_heads=n_heads
            )
        else:
            self.gat = nn.Linear(node_in_dim, hidden_dim)

        if self.use_ode:
            self.ode_rnn = ODERNNCell(
                input_dim=hidden_dim, hidden_dim=hidden_dim, solver='rk4'
            )
        else:
            self.ode_rnn = nn.GRUCell(hidden_dim, hidden_dim)

        self.message_passer = TopologyMessagePasser(hidden_dim)
        self.actor = IndependentActionHeads(hidden_dim=hidden_dim)
        self.critic = CentralizedNoiselessCritic(
            global_state_dim=global_in_dim, hidden_dim=hidden_dim
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.config['model']['hidden_dim'], device=device
        )

    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        node_obs = torch.zeros(batch_size, 100, 256, device=obs.device)
        node_obs[:, 0, :] = obs
        return node_obs

    def select_action(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
        dt: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        siem_emb = kwargs.get('siem_embedding')
        adj_mask = kwargs.get('adj_mask')
        if adj_mask is None:
            from src.models.ct_gmarl.gat_processor import SubnetMaskGenerator

            adj_mask = SubnetMaskGenerator.create_mask(num_nodes=100).to(obs.device)

        # SIEM-Fused Observation
        node_obs = self._preprocess_obs(obs)
        if siem_emb is not None:
            # Inject SIEM vector into Gateway node (0) reasoning
            if siem_emb.dim() == 2:
                siem_emb = siem_emb.unsqueeze(1)
            node_obs[:, 0, :128] += siem_emb[:, 0, :]  # Fusion at head node

        if adj_mask.dim() == 2:
            adj_mask = adj_mask.unsqueeze(0)

        if self.use_gat:
            spatial_feats = self.gat(node_obs, adj_mask)
            pooled_feat = torch.mean(spatial_feats, dim=1)
        else:
            # Bypass GNN: Global Mean Pool of node observations
            pooled_feat = torch.mean(self.gat(node_obs), dim=1)

        if self.use_ode:
            h_new, nfe = self.ode_rnn(pooled_feat, h_prev, dt)
        else:
            # Bypass ODE: Standard GRU update (Ignores dt)
            h_new = self.ode_rnn(pooled_feat, h_prev)
            nfe = 0

        # Cross-agent message passing
        agent_id = kwargs.get('agent_id')
        if agent_id:
            # We assume a shared dictionary is passed in kwargs during rollout
            shared_h = kwargs.get('shared_hidden_states', {})
            shared_h[agent_id] = h_new
            self.message_passer(shared_h)
            h_new = shared_h.get(agent_id, h_new)

        logits_type, logits_target = self.actor(h_new, mask)
        dist_type = functional.softmax(logits_type, dim=-1)
        dist_target = functional.softmax(logits_target, dim=-1)

        a_type = torch.multinomial(dist_type, 1).squeeze(-1)
        a_target = torch.multinomial(dist_target, 1).squeeze(-1)

        # Clear the reference after processing to avoid holding onto computation graphs
        if agent_id:
            shared_h[agent_id] = h_new.detach()

        log_prob = torch.log(
            dist_type.gather(1, a_type.unsqueeze(-1)).squeeze(-1) + 1e-10
        ) + torch.log(dist_target.gather(1, a_target.unsqueeze(-1)).squeeze(-1) + 1e-10)

        return torch.stack([a_type, a_target], dim=-1), log_prob, h_new, {'nfe': nfe}

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
        dt: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        adj_mask = kwargs.get('adj_mask')
        if adj_mask is None:
            from src.models.ct_gmarl.gat_processor import SubnetMaskGenerator

            adj_mask = SubnetMaskGenerator.create_mask(num_nodes=100).to(obs.device)

        siem_emb = kwargs.get('siem_embedding')
        node_obs = self._preprocess_obs(obs)
        if siem_emb is not None:
            if siem_emb.dim() == 2:
                siem_emb = siem_emb.unsqueeze(1)
            node_obs[:, 0, :128] += siem_emb[:, 0, :]

        if adj_mask.dim() == 2:
            adj_mask = adj_mask.unsqueeze(0)

        if self.use_gat:
            spatial_feats = self.gat(node_obs, adj_mask)
            pooled_feat = torch.mean(spatial_feats, dim=1)
        else:
            pooled_feat = torch.mean(self.gat(node_obs), dim=1)

        if self.use_ode:
            h_new, nfe = self.ode_rnn(pooled_feat, h_prev, dt)
        else:
            h_new = self.ode_rnn(pooled_feat, h_prev)
            nfe = 0

        # Cross-agent message passing during evaluation
        agent_id = kwargs.get('agent_id')
        if agent_id:
            shared_h = kwargs.get('shared_hidden_states', {})
            shared_h[agent_id] = h_new
            self.message_passer(shared_h)
            h_new = shared_h.get(agent_id, h_new)

        logits_type, logits_target = self.actor(h_new, mask)
        dist_type = functional.softmax(logits_type, dim=-1)
        dist_target = functional.softmax(logits_target, dim=-1)

        lp_type = torch.log(
            dist_type.gather(1, actions[:, 0].unsqueeze(-1)).squeeze(-1) + 1e-10
        )
        lp_target = torch.log(
            dist_target.gather(1, actions[:, 1].unsqueeze(-1)).squeeze(-1) + 1e-10
        )

        ent_type = -torch.sum(dist_type * torch.log(dist_type + 1e-10), dim=-1)
        ent_target = -torch.sum(dist_target * torch.log(dist_target + 1e-10), dim=-1)

        return (
            (lp_type + lp_target).unsqueeze(-1),
            (ent_type + ent_target).unsqueeze(-1),
            {'nfe': nfe},
        )

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.critic(global_state)
