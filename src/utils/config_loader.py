from typing import Optional

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


class ConfigLoader:
    """
    This class provides a interface to Hydra's
    composition engine, allowing configuration resolution without
    relying on the @hydra.main() decorator. This prevents issues
    with global state in complex MARL training loops.
    """

    def __init__(self, config_path: str = '../../conf', config_name: str = 'config'):
        """
        Initializes the Hydra environment.

        Args:
            config_path (str): Relative path from the script to the 'conf' folder.
            config_name (str): The primary YAML file name (without extension).
        """
        self.config_path = config_path
        self.config_name = config_name
        self._initialize_hydra()

    def _initialize_hydra(self) -> None:
        """Sets up the Hydra context and global search paths."""
        # Using the global initialize context (safely handles re-initialization)
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            initialize(config_path=self.config_path, version_base=None)
            print(
                f'[ConfigLoader] Hydra initialized with search path: {self.config_path}'
            )

    def load_config(self, overrides: Optional[list] = None) -> DictConfig:
        """
        Composes and resolves the full hierarchical configuration.

        Args:
            overrides (Optional[list]): List of CLI-style overrides (e.g., ['seed=101']).

        Returns:
            DictConfig: Fully resolved OmegaConf object.
        """
        cfg = compose(config_name=self.config_name, overrides=overrides or [])
        # Resolving interpolation (e.g., ${model.name})
        OmegaConf.resolve(cfg)
        return cfg

    @staticmethod
    def print_config(cfg: DictConfig) -> None:
        """Pretty-prints the resolved configuration for logging."""
        print('-' * 40)
        print('Resolved CT-GMARL Configuration')
        print('-' * 40)
        print(OmegaConf.to_yaml(cfg))
        print('-' * 40)
