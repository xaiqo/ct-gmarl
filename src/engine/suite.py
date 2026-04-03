from typing import Any

import torch

from src.engine.buffer import POSMDPBuffer
from src.engine.manager import ForgeAgentManager
from src.engine.optimizer import ForgeOptimizationHead
from src.engine.runner import ForgeRolloutRunner
from src.engine.telemetry import ForgeTelemetryManager
from src.utils.logger import WandBLogger


class ForgeSuite:
    """
    Unifies the manager, runner, optimizer, and telemetry into a single,
    training loop.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize W&B Telemetry
        run_name = cfg.get('timestamp', 'NeuralForge_Run')
        self.logger = WandBLogger(cfg, name=run_name) if not cfg.smoke_test else None

        # 1. Initialize Management
        from netforge_rl.environment.parallel_env import NetForgeRLEnv

        self.env = NetForgeRLEnv(cfg.env)
        self.manager = ForgeAgentManager(cfg, self.env.possible_agents, self.device)

        # 2. Initialize Memories
        self.blue_buffer = self._init_buffer(
            self.manager.blue_agents, cfg.env.get('obs_dim', 256)
        )
        self.red_buffer = self._init_buffer(
            self.manager.red_agents, cfg.env.get('obs_dim', 256)
        )
        self.manager.blue_buffer = self.blue_buffer
        self.manager.red_buffer = self.red_buffer

        # 3. Initialize Orchestrators
        self.runner = ForgeRolloutRunner(
            cfg, self.env, self.manager, self.blue_buffer, self.red_buffer, self.device
        )
        self.optimizer = ForgeOptimizationHead(cfg, self.manager, self.device)
        self.telemetry = ForgeTelemetryManager(cfg, self.logger)

    def _init_buffer(self, agents, obs_dim):
        return POSMDPBuffer(
            capacity=self.cfg.buffer_size,
            num_agents=len(agents),
            obs_shape=(obs_dim,),
            hidden_dim=self.cfg.model.hidden_dim,
            global_state_dim=512,
        )

    def train(self):
        """Main training loop orchestrating rollouts and updates."""
        print(
            f'[NeuralForge] Launching Competition: {self.cfg.blue_algorithm} vs {self.cfg.red_algorithm}'
        )

        for _ in range(self.cfg.episodes):
            # Rollout Phase
            stats = self.runner.run_episode()

            # Optimization Phase
            t_blue = self.optimizer.update_team('blue') if stats['steps'] > 0 else {}
            t_red = (
                self.optimizer.update_team('red')
                if (self.manager.red_learning and stats['steps'] > 0)
                else {}
            )

            # Telemetry Phase
            self.telemetry.finalize_metrics(stats, t_blue, t_red, self.env)

            # Buffer Cleanup
            self.blue_buffer.clear()
            self.red_buffer.clear()

    def cleanup(self):
        if self.logger:
            self.logger.finish()
