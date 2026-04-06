import time
from typing import Any, Dict

import torch

from src.engine.manager import ForgeAgentManager


class ForgeRolloutRunner:
    """
    Asynchronous Rollout Orchestrator for Competitive MARL.

    Handles environment step loops, trajectory collection, and budget tracking.
    """

    def __init__(
        self,
        cfg: Any,
        env: Any,
        manager: ForgeAgentManager,
        blue_buffer: Any,
        red_buffer: Any,
        device: torch.device,
    ):
        self.cfg = cfg
        self.env = env
        self.manager = manager
        self.blue_buffer = blue_buffer
        self.red_buffer = red_buffer
        self.device = device

    def run_episode(self) -> Dict[str, Any]:
        """Executes a single high-fidelity episode and collects trajectories."""
        obs_dict, _ = self.env.reset()
        h_blue, h_red = self.manager.init_hidden()
        blue_agent, red_agent = self.manager.get_agents()

        agent_ep_rewards = {aid: 0.0 for aid in self.env.possible_agents}
        agent_ep_actions = {aid: [] for aid in self.env.possible_agents}
        agent_ep_targets = {aid: [] for aid in self.env.possible_agents}
        ep_security_stats = {
            'false_positives': 0.0,
            'successful_exploits': 0.0,
            'services_restored': 0.0,
            'hosts_isolated': 0.0,
            'SLA_Uptime_Percentage': 1.0,
            'MTTC': 0.0,
            'Red_Dwell_Time': 0.0,
            'Total_Exfiltrated_Data': 0.0,
        }

        total_nfe_blue, total_nfe_red = 0.0, 0.0
        total_delta_t = 0.0
        ep_steps, done = 0, False
        start_time = time.time()

        while not done:
            gt = self._get_gt()
            actions_dict, log_probs = {}, {}

            # ACTION SAMPLING
            actions_dict, log_probs, nfe_b, nfe_r = self._sample_actions(
                obs_dict, h_blue, h_red, blue_agent, red_agent
            )
            total_nfe_blue += nfe_b
            total_nfe_red += nfe_r

            # STEP ENV
            next_obs_dict, rewards, terminated, truncated, s_info = self.env.step(
                actions_dict
            )
            next_gt = self._get_gt()

            # BUFFER INSERTION
            self._insert_trajectories(
                obs_dict,
                next_obs_dict,
                h_blue,
                h_red,
                actions_dict,
                rewards,
                log_probs,
                gt,
                next_gt,
                terminated,
                truncated,
            )

            # METRIC TRACKING
            self._track_stats(
                actions_dict,
                rewards,
                s_info,
                agent_ep_rewards,
                agent_ep_actions,
                agent_ep_targets,
                ep_security_stats,
            )

            obs_dict = next_obs_dict
            ep_steps += 1
            done = any(terminated.values()) or any(truncated.values())

        return {
            'rewards': agent_ep_rewards,
            'actions': agent_ep_actions,
            'steps': ep_steps,
            'nfe_blue': total_nfe_blue / (ep_steps * len(self.manager.blue_agents))
            if ep_steps > 0
            else 0.0,
            'nfe_red': total_nfe_red / (ep_steps * len(self.manager.red_agents))
            if ep_steps > 0
            else 0.0,
            'delta_t': total_delta_t / ep_steps if ep_steps > 0 else 0.0,
            'security': s_info,
            'ep_security': ep_security_stats,
            'ep_targets': agent_ep_targets,
            'duration': time.time() - start_time,
        }

    def _sample_actions(self, obs_dict, h_blue, h_red, blue_agent, red_agent):
        actions_dict, log_probs = {}, {}
        nfe_b, nfe_r = 0.0, 0.0

        shared_blue_h = {}

        # Enforce strict topology ordering: DMZ -> Internal -> Restricted
        blue_eval_order = []
        if 'blue_dmz' in self.manager.blue_agents:
            blue_eval_order.append('blue_dmz')
        if 'blue_internal' in self.manager.blue_agents:
            blue_eval_order.append('blue_internal')
        if 'blue_restricted' in self.manager.blue_agents:
            blue_eval_order.append('blue_restricted')

        for aid in [a for a in self.manager.blue_agents if a not in blue_eval_order]:
            blue_eval_order.append(aid)

        for aid in blue_eval_order:
            obs_raw = obs_dict[aid]
            obs_t, dt_t, mask_t, siem_t, adj_t = self._to_tensor(obs_raw)
            a, lp, h, ex = blue_agent.select_action(
                obs_t,
                h_blue[aid],
                dt_t,
                mask_t,
                siem_embedding=siem_t,
                adj_mask=adj_t,
                agent_id=aid,
                shared_hidden_states=shared_blue_h,
            )
            actions_dict[aid], log_probs[aid], h_blue[aid] = (
                [int(a[0, 0]), int(a[0, 1])],
                lp,
                h.detach(),
            )
            nfe_b += ex.get('nfe', 1.0)

        for aid in self.manager.red_agents:
            obs_raw = obs_dict[aid]
            obs_t, dt_t, mask_t, siem_t, adj_t = self._to_tensor(obs_raw)
            a, lp, h, ex = red_agent.select_action(
                obs_t, h_red[aid], dt_t, mask_t, siem_embedding=siem_t, adj_mask=adj_t
            )
            actions_dict[aid], log_probs[aid], h_red[aid] = (
                [int(a[0, 0]), int(a[0, 1])],
                lp,
                h.detach(),
            )
            nfe_r += ex.get('nfe', 1.0)

        return actions_dict, log_probs, nfe_b, nfe_r

    def _to_tensor(self, obs_raw):
        o = torch.from_numpy(obs_raw['obs']).float().to(self.device).unsqueeze(0)
        dt = torch.from_numpy(obs_raw['delta_t']).float().to(self.device).unsqueeze(0)
        m = (
            torch.from_numpy(obs_raw['action_mask'])
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        s = (
            torch.from_numpy(obs_raw['siem_embedding'])
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        adj = (
            torch.from_numpy(obs_raw['adj_matrix'])
            .reshape(100, 100)
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        return o, dt, m, s, adj

    def _get_gt(self):
        return (
            torch.from_numpy(self.env.global_state_vector())
            .float()
            .to(self.device)
            .unsqueeze(0)
        )

    def _insert_trajectories(
        self, o_d, no_d, h_b, h_r, a_d, r, lp, gt, ngt, term, trunc
    ):
        d_val = float(any(term.values()) or any(trunc.values()))

        def insert_team(agents, buffer, h_p):
            obs = torch.stack(
                [torch.from_numpy(o_d[aid]['obs']).float() for aid in agents]
            )
            h = torch.stack(
                [
                    h_p[aid].squeeze(0)
                    if not isinstance(h_p[aid], tuple)
                    else h_p[aid][0].squeeze(0)
                    for aid in agents
                ]
            )
            dt = torch.stack(
                [torch.from_numpy(o_d[aid]['delta_t']).float() for aid in agents]
            )
            acts = torch.stack([torch.tensor(a_d[aid]) for aid in agents])
            rews = torch.stack([torch.tensor([r[aid]]) for aid in agents])
            masks = torch.stack(
                [torch.from_numpy(o_d[aid]['action_mask']).float() for aid in agents]
            )
            probs = torch.stack(
                [
                    lp[aid].detach() if aid in lp else torch.tensor([0.0])
                    for aid in agents
                ]
            )
            n_obs = torch.stack(
                [torch.from_numpy(no_d[aid]['obs']).float() for aid in agents]
            )
            siem = torch.stack(
                [torch.from_numpy(o_d[aid]['siem_embedding']).float() for aid in agents]
            )
            adj = torch.stack(
                [
                    torch.from_numpy(o_d[aid]['adj_matrix']).float().reshape(100, 100)
                    for aid in agents
                ]
            )
            d_t = torch.tensor([[d_val]] * len(agents))
            buffer.insert(
                obs,
                h,
                torch.zeros_like(h),
                dt,
                acts,
                rews,
                masks,
                gt.squeeze(0),
                n_obs,
                ngt.squeeze(0),
                d_t,
                probs,
                siem,
                adj_matrix=adj,
            )

        insert_team(self.manager.blue_agents, self.blue_buffer, h_b)
        if self.manager.red_learning:
            insert_team(self.manager.red_agents, self.red_buffer, h_r)

    def _track_stats(self, a_d, r, s_info, ep_r, ep_a, ep_t, ep_s):
        for aid, rew in r.items():
            ep_r[aid] += rew
        for aid, acts in a_d.items():
            ep_a[aid].append(acts[0])
        for aid, ainfo in s_info.items():
            if 'red' in aid.lower() and ainfo.get('successful_exploits'):
                ep_s['successful_exploits'] += ainfo['successful_exploits']
            if 'blue' in aid.lower():
                for k in ['false_positives', 'services_restored', 'hosts_isolated']:
                    ep_s[k] += ainfo.get(k, 0.0)

            # Continuous metric tracking (pulling from last info dict)
            for k in [
                'SLA_Uptime_Percentage',
                'MTTC',
                'Red_Dwell_Time',
                'Total_Exfiltrated_Data',
            ]:
                if k in ainfo:
                    ep_s[k] = ainfo[k]

            if ainfo.get('target_ip_index') is not None:
                ep_t[aid].append(ainfo['target_ip_index'])
