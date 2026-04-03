from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


class ForgeOptimizationHead:
    """
    Modular Optimization Engine for Multi-Agent RL.

    Handles GAE calculation, PPO/QMIX gradient passes, and epoch updates.
    """

    def __init__(self, cfg: Any, manager: Any, device: torch.device):
        self.cfg = cfg
        self.manager = manager
        self.device = device
        self.eps_clip = 0.2
        self.beta = cfg.get('beta', 0.05)
        self.gae_lambda = 0.95

    def update_team(self, team='blue'):
        """Performs a full PPO update epoch for the specified team."""
        agent, optimizer = (
            (self.manager.blue_agent, self.manager.blue_optimizer)
            if team == 'blue'
            else (self.manager.red_agent, self.manager.red_optimizer)
        )
        buffer = self.manager.blue_buffer if team == 'blue' else self.manager.red_buffer

        self._calculate_advantages(agent, buffer)

        train_stats = []
        for _ in range(self.cfg.get('ppo_epochs', 10)):
            batch = buffer.sample(self.cfg.batch_size)
            if not batch:
                continue

            stats = self._ppo_step(agent, optimizer, batch)
            train_stats.append(stats)

        return (
            {k: np.mean([s[k] for s in train_stats]) for k in train_stats[0].keys()}
            if train_stats
            else {}
        )

    def _calculate_advantages(self, agent, buffer):
        with torch.no_grad():
            upper = buffer.capacity if buffer.is_full else buffer.ptr
            vals = agent.get_value(buffer.global_states[:upper].to(self.device)).cpu()
            next_vals = agent.get_value(
                buffer.next_global_states[:upper].to(self.device)
            ).cpu()

            advs = torch.zeros_like(buffer.rewards[:upper])
            gae = 0
            for t in reversed(range(upper)):
                dt = buffer.delta_ts[t]
                discount = torch.exp(-self.beta * dt)
                delta = (
                    buffer.rewards[t]
                    + discount * next_vals[t] * (1.0 - buffer.dones[t])
                    - vals[t]
                )
                gae = delta + discount * self.gae_lambda * (1.0 - buffer.dones[t]) * gae
                advs[t] = gae
            buffer.advantages[:upper] = advs
            buffer.returns[:upper] = advs + vals.unsqueeze(1)

    def _ppo_step(self, agent, optimizer, batch):
        obs, h_p, dt, acts, masks, old_lp, advs, rets, gs, siem = [
            batch[k].to(self.device)
            for k in [
                'obs',
                'h_prev',
                'dt',
                'actions',
                'masks',
                'log_probs',
                'advantages',
                'returns',
                'global_state',
                'siem_emb',
            ]
        ]
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        new_lp, entropy, _ = agent.evaluate_actions(
            obs.view(-1, *obs.shape[2:]),
            h_p.view(-1, *h_p.shape[2:]),
            dt.view(-1, *dt.shape[2:]),
            acts.view(-1, *acts.shape[2:]),
            masks.view(-1, *masks.shape[2:]),
            siem_embedding=siem.view(-1, *siem.shape[2:]),
        )

        ratio = torch.exp(new_lp - old_lp.view(-1, 1))
        surr1 = ratio * advs.view(-1, 1)
        surr2 = torch.clamp(
            ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip
        ) * advs.view(-1, 1)
        p_loss = -torch.min(surr1, surr2).mean()

        v_curr = agent.get_value(gs)
        v_loss = functional.mse_loss(v_curr, rets.mean(dim=1))

        loss = p_loss + 0.5 * v_loss - 0.01 * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 10.0)
        optimizer.step()

        return {
            'p_loss': p_loss.item(),
            'v_loss': v_loss.item(),
            'entropy': entropy.mean().item(),
            'kl': (old_lp.view(-1, 1) - new_lp).mean().item(),
            'grad_norm': grad_norm.item(),
        }
