from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb


class WandBLogger:
    """
    This class handles the Weights & Biases (W&B) lifecycle, ensuring
    tracking of asynchronous MARL metrics (SLA Uptime, MTTC) and implementing
    state recovery via online artifacts.
    """

    def __init__(
        self,
        config: Any,
        project: str = 'ct-gmarl-research',
        mode: str = 'online',
        resume: str = 'allow',
        name: Optional[str] = None,
        job_type: Optional[str] = None,
    ):
        """
        Initializes the W&B run with configuration metadata and numerical naming.
        """
        from omegaconf import OmegaConf

        # Force conversion to a standard Python dict with full resolution of all Hydra variables.
        sanitized_config = (
            OmegaConf.to_container(config, resolve=True)
            if not isinstance(config, dict)
            else config
        )

        self.run = wandb.init(
            project=project,
            config=sanitized_config,
            mode=mode,
            resume=resume,
            name=name,
            job_type=job_type,
        )
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        print(f'[WandBLogger] Initialized run: {self.run.name} ({self.run.id})')

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Logs numerical metrics to the W&B dashboard for real-time visualization.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics (e.g., 'sla_uptime', 'mttc').
            step (Optional[int]): Current environment tick or step index.
        """
        wandb.log(metrics, step=step)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        tick: int,
        filename: str = 'latest_checkpoint.pt',
    ) -> str:
        """
        Saves a full state recovery bundle locally and as a W&B Artifact.

        Args:
            model (torch.nn.Module): The policy or model state_dict.
            optimizer (torch.optim.Optimizer): Optimizer state for resuming.
            scheduler (Any): Learning rate scheduler state.
            tick (int): The current simulated environment tick.
            filename (str): Name for the local file.

        Returns:
            str: Path to the local checkpoint file.
        """
        path = self.checkpoint_dir / filename
        state = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'tick': tick,
        }
        torch.save(state, path)

        # Log as an online artifact for Sim2Real reproducibility
        artifact = wandb.Artifact(
            name=f'model-checkpoint-{self.run.id}',
            type='model',
            description=f'State recovery bundle for tick {tick}',
        )
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

        print(f'[WandBLogger] Robust checkpoint saved: {path} (Tick {tick})')
        return str(path)

    def load_checkpoint(
        self,
        artifact_name: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> int:
        """
        Resumes training state from a remote W&B artifact.

        Args:
            artifact_name (str): Full artifact identifier (e.g. 'entity/project/name:alias').
            model (torch.nn.Module): Model to load weights into.
            optimizer (Optional[torch.optim.Optimizer]): Optimizer state to recover.
            scheduler (Optional[Any]): Scheduler state to recover.

        Returns:
            int: The tick at which the environment should resume.
        """
        artifact = self.run.use_artifact(artifact_name)
        artifact_dir = artifact.download()

        # Assume standard filename from save_checkpoint
        checkpoint_path = Path(artifact_dir) / 'latest_checkpoint.pt'
        state = torch.load(checkpoint_path)

        model.load_state_dict(state['model_state'])
        if optimizer:
            optimizer.load_state_dict(state['optimizer_state'])
        if scheduler and state['scheduler_state']:
            scheduler.load_state_dict(state['scheduler_state'])

        print(
            f'[WandBLogger] Successfully resumed state from artifact: {artifact_name}'
        )
        return state['tick']

    def finish(self) -> None:
        """Gracefully closes the W&B session."""
        self.run.finish()
