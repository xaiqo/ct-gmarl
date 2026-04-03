import random

import numpy as np
import torch


class SeedManager:
    """
    Centralized manager for deterministic research reproducibility.

    This class ensures that all stochastic components of the MARL pipeline
    (Neural Networks, Environment, and Python built-ins) are frozen to a
    specific seed, enabling exact replication of research results.
    """

    @staticmethod
    def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
        """
        Sets the seed for all relevant libraries and configures CuDNN.

        Args:
            seed (int): The seed value to use across all frameworks.
            deterministic_cudnn (bool): If True, forces CuDNN to use deterministic
                algorithms. Note: This may impact performance.
        """
        # 1. Python built-in random
        random.seed(seed)

        # 2. NumPy
        np.random.seed(seed)

        # 3. PyTorch (CPU and all GPUs)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 4. CuDNN Determinism
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f'[SeedManager] Statistical randomness frozen at seed: {seed}')

    @staticmethod
    def get_seed_from_config(config: dict) -> int:
        """
        Extracts seed from a configuration dictionary, defaults to 42.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            int: The extracted or default seed.
        """
        return config.get('seed', 42)
