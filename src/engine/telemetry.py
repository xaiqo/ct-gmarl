from typing import Any, Dict

import numpy as np
import wandb

from src.utils.logger import WandBLogger


class ForgeTelemetryManager:
    """
    Standardized Telemetry Engine for Research Metrics.

    Handles metric accumulation, WandB orchestration, and high-fidelity
    security dashboards (Fiscal, Security, POSMDP).
    """

    def __init__(self, cfg: Any, logger: WandBLogger = None):
        self.cfg = cfg
        self.logger = logger

    def finalize_metrics(
        self,
        s: Dict[str, Any],
        t_blue: Dict[str, float],
        t_red: Dict[str, float],
        env: Any,
    ):
        """Processes and logs the complete metric suite to WandB."""
        blue_agents = [aid for aid in env.possible_agents if 'blue' in aid.lower()]
        red_agents = [aid for aid in env.possible_agents if 'red' in aid.lower()]
        rep_aid = blue_agents[0]
        sec = s['security'].get(rep_aid, {})

        stats = {
            'Blue_Team/Aggregate_Reward': float(
                sum(r for aid, r in s['rewards'].items() if 'blue' in aid.lower())
            ),
            'Red_Team/Aggregate_Reward': float(
                sum(r for aid, r in s['rewards'].items() if 'red' in aid.lower())
            ),
            'Business/SLA_Uptime_Percentage': float(
                sec.get('SLA_Uptime_Percentage', 0.0)
            ),
            'Security/MTTC': float(sec.get('MTTC', 0.0)),
            'Security/Total_Exfiltrated_Data': float(
                sec.get('Total_Exfiltrated_Data', 0.0)
            ),
            'POSMDP/Average_Delta_T': float(s['delta_t']),
            'POSMDP/ODE_NFE': float(s['nfe_blue']),
            'System/Steps_Per_Second': float(s['steps']) / float(s['duration'])
            if s['duration'] > 0
            else 0.0,
        }

        # Per-Agent Granular Rewards
        for aid, rew in s['rewards'].items():
            stats[f'Agent/{aid}/Reward'] = float(rew)

        # Targeted Metrics Histograms
        for aid, acts in s['actions'].items():
            label = (
                'red_actions'
                if 'red' in aid.lower()
                else f'blue_actions_{aid.split("_")[-1]}'
            )
            if acts:
                stats[label] = wandb.Histogram(np.array(acts).tolist())

        for aid, targs in s.get('ep_targets', {}).items():
            label = (
                'red_targets'
                if 'red' in aid.lower()
                else f'blue_targets_{aid.split("_")[-1]}'
            )
            if targs:
                stats[label] = wandb.Histogram(np.array(targs).tolist())

        # Security KPI Synthesis
        ep_sec = s.get('ep_security', {})
        stats.update(
            {
                'Security/Total_False_Positives': float(
                    ep_sec.get('false_positives', 0)
                ),
                'Security/Total_Successful_Exploits': float(
                    ep_sec.get('successful_exploits', 0)
                ),
                'Security/Total_Services_Restored': float(
                    ep_sec.get('services_restored', 0)
                ),
                'Security/Infrastructural_Isolation_Count': float(
                    ep_sec.get('hosts_isolated', 0)
                ),
            }
        )

        # Economic Dashboard Synthesis
        blue_rep, red_rep = blue_agents[0], red_agents[0]
        stats.update(
            {
                'Economy/Blue_Sovereign_Funds': float(
                    env.global_state.agent_funds.get(blue_rep, 0)
                ),
                'Economy/Red_Shadow_Funds': float(
                    env.global_state.agent_funds.get(red_rep, 0)
                ),
                'Economy/Blue_Compute_Utilization': 1.0
                - (float(env.global_state.agent_compute.get(blue_rep, 0)) / 1000.0),
                'Economy/Red_Compute_Utilization': 1.0
                - (float(env.global_state.agent_compute.get(red_rep, 0)) / 1000.0),
            }
        )

        # Identity Telemetry
        red_inventory = env.global_state.agent_inventory.get(red_agents[0], set())
        stats['ZeroTrust/Enterprise_Admin_Token_Acquired'] = (
            1.0 if 'Enterprise_Admin_Token' in red_inventory else 0.0
        )

        if t_blue:
            stats.update({f'Blue_Training/{k}': float(v) for k, v in t_blue.items()})
        if t_red:
            stats.update({f'Red_Training/{k}': float(v) for k, v in t_red.items()})

        if self.logger:
            self.logger.log_metrics(stats)
