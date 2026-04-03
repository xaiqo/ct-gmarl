from typing import Any, List

import torch
from omegaconf import OmegaConf

from src.models.factory import ModelFactory


class ForgeAgentManager:
    """
    Modular interface for managing competing Multi-Agent teams (Blue/Red).

    Handles model instantiation, hidden state initialization, and hardware mapping.
    """

    def __init__(self, cfg: Any, possible_agents: List[str], device: torch.device):
        self.cfg = cfg
        self.device = device

        self.blue_agents = [aid for aid in possible_agents if 'blue' in aid.lower()]
        self.red_agents = [aid for aid in possible_agents if 'red' in aid.lower()]

        full_cfg = OmegaConf.to_container(cfg, resolve=True)

        self.blue_agent = ModelFactory.create(cfg.blue_algorithm, full_cfg).to(device)
        self.blue_optimizer = torch.optim.Adam(self.blue_agent.parameters(), lr=cfg.lr)

        self.red_learning = True
        self.red_agent = ModelFactory.create(cfg.red_algorithm, full_cfg).to(device)
        self.red_optimizer = torch.optim.Adam(self.red_agent.parameters(), lr=cfg.lr)

    def init_hidden(self):
        """Initializes hidden states for all agents in the competition."""
        h_blue = {
            aid: self.blue_agent.init_hidden(1, self.device) for aid in self.blue_agents
        }
        h_red = {
            aid: self.red_agent.init_hidden(1, self.device) for aid in self.red_agents
        }
        return h_blue, h_red

    def get_optimizers(self):
        return self.blue_optimizer, self.red_optimizer

    def get_agents(self):
        return self.blue_agent, self.red_agent
