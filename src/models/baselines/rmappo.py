from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.models.base import BaseAgent


class MLPActionHeads(nn.Module):
    def __init__(self, hidden_dim: int, n_types: int = 12, n_targets: int = 50):
        super(MLPActionHeads, self).__init__()
        self.policy_type = nn.Linear(hidden_dim, n_types)
        self.policy_target = nn.Linear(hidden_dim, n_targets)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_type = mask[:, :12]
        mask_target = mask[:, 12:]
        logits_type = self.policy_type(h)
        logits_target = self.policy_target(h)
        inf_mask_type = (1.0 - mask_type) * -1e9
        inf_mask_target = (1.0 - mask_target) * -1e9
        return logits_type + inf_mask_type, logits_target + inf_mask_target


class RMAPPOAgent(BaseAgent):
    """
    Standard Recurrent MAPPO Baseline.

    Uses LSTM to process observation sequences, providing a discrete-time
    alternative to the continuous-time CT-GMARL architecture.
    """

    def __init__(self, config: Dict[str, Any]):
        super(RMAPPOAgent, self).__init__(config)
        hidden_dim = config['model']['hidden_dim']
        obs_dim = config.get('obs_dim', 256)
        global_in_dim = config.get('global_state_dim', 512)

        # Discrete temporal processing (LSTM)
        self.lstm = nn.LSTMCell(obs_dim, hidden_dim)
        self.actor = MLPActionHeads(hidden_dim=hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(global_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """LSTM requires both hidden (h) and cell (c) states."""
        dim = self.config['model']['hidden_dim']
        h = torch.zeros(batch_size, dim, device=device)
        c = torch.zeros(batch_size, dim, device=device)
        return (h, c)

    def select_action(
        self,
        obs: torch.Tensor,
        h_prev: Tuple[torch.Tensor, torch.Tensor],
        dt: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]
    ]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Standard LSTM update (ignores continuous dt)
        h_new = self.lstm(obs, h_prev)

        # Policy uses hidden state h (h_new[0])
        logits_type, logits_target = self.actor(h_new[0], mask)
        dist_type = F.softmax(logits_type, dim=-1)
        dist_target = F.softmax(logits_target, dim=-1)

        a_type = torch.multinomial(dist_type, 1).squeeze(-1)
        a_target = torch.multinomial(dist_target, 1).squeeze(-1)

        log_prob = torch.log(
            dist_type.gather(1, a_type.unsqueeze(-1)).squeeze(-1)
        ) + torch.log(dist_target.gather(1, a_target.unsqueeze(-1)).squeeze(-1))

        return torch.stack([a_type, a_target], dim=-1), log_prob, h_new, {}

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        h_prev: Tuple[torch.Tensor, torch.Tensor],
        dt: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h_new = self.lstm(obs, h_prev)

        logits_type, logits_target = self.actor(h_new[0], mask)
        dist_type = F.softmax(logits_type, dim=-1)
        dist_target = F.softmax(logits_target, dim=-1)

        lp_type = torch.log(
            dist_type.gather(1, actions[:, 0].unsqueeze(-1)).squeeze(-1)
        )
        lp_target = torch.log(
            dist_target.gather(1, actions[:, 1].unsqueeze(-1)).squeeze(-1)
        )

        ent_type = -torch.sum(dist_type * torch.log(dist_type + 1e-10), dim=-1)
        ent_target = -torch.sum(dist_target * torch.log(dist_target + 1e-10), dim=-1)

        return (
            (lp_type + lp_target).unsqueeze(-1),
            (ent_type + ent_target).unsqueeze(-1),
            {},
        )

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.critic(global_state)
