"""PyTorch Checkpoint Manager for saving, loading, and restoring RL agent state."""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from src.agents.dqn.agent import SnakeDQNAgent


class CheckpointManager:
    """Reusable PyTorch Checkpoint Manager for saving/restoring training checkpoints and best model deliverables."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "checkpoints/snake",
        format_version: str = "1.0",
    ):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Path to directory where periodic checkpoints are saved (default: 'checkpoints/snake').
            format_version: Version identifier string for saved checkpoint format.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.format_version = format_version
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        agent: SnakeDQNAgent,
        episode: int,
        filename: Optional[str] = None,
        is_best: bool = False,
        best_filename: Union[str, Path] = "models/snake_dqn_best.pt",
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save structured checkpoint file atomically.

        Args:
            agent: SnakeDQNAgent instance to save.
            episode: Current episode number.
            filename: Optional explicit filename (e.g. 'checkpoint_ep100.pt'). Default: 'checkpoint_ep<episode>.pt'.
            is_best: If True, also save a copy to best_filename export path.
            best_filename: Path for best model export (default: 'models/snake_dqn_best.pt').
            extra_info: Optional extra metadata to store in checkpoint.

        Returns:
            Path object pointing to the saved checkpoint file.
        """
        if filename is None:
            filename = f"checkpoint_ep{episode}.pt"

        target_path = self.checkpoint_dir / filename

        # Collect RNG states
        rng_state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass

        checkpoint_data: Dict[str, Any] = {
            "format_version": self.format_version,
            "agent_state": agent.get_checkpoint_state(),
            "episode": int(episode),
            "frame_count": int(agent.frame_count),
            "update_steps": int(agent.update_steps),
            "epsilon": float(agent.get_epsilon()),
            "config": agent.config,
            "rng_state": rng_state,
            "extra_info": extra_info or {},
        }

        # Atomic safe-save: write to temporary file first, then replace target
        temp_path = target_path.with_suffix(".pt.tmp")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_data, temp_path)
        temp_path.replace(target_path)

        if is_best:
            best_path = Path(best_filename)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            best_temp = best_path.with_suffix(".pt.tmp")
            torch.save(checkpoint_data, best_temp)
            best_temp.replace(best_path)

        return target_path

    def load(
        self,
        checkpoint_path: Union[str, Path],
        agent: Optional[SnakeDQNAgent] = None,
        map_location: Union[str, torch.device] = "cpu",
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint file and optionally restore state into an agent instance.

        Args:
            checkpoint_path: Path to checkpoint file.
            agent: Optional SnakeDQNAgent instance into which parameters will be loaded.
            map_location: Device mapping target for PyTorch load (default: 'cpu').
            restore_rng: Whether to restore random number generator states if available.

        Returns:
            Dict containing complete loaded checkpoint data.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            ValueError: If file is not a valid checkpoint.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        try:
            data = torch.load(path, map_location=map_location, weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch checkpoint file at {path}: {e}")

        if not isinstance(data, dict) or "format_version" not in data:
            raise ValueError(f"Invalid checkpoint format at {path}: missing format_version.")

        if agent is not None:
            agent_state = data.get("agent_state", {})
            agent.load_checkpoint_state(agent_state)

        if restore_rng and "rng_state" in data:
            rng = data["rng_state"]
            if "python" in rng:
                random.setstate(rng["python"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch_cpu" in rng:
                torch.set_rng_state(rng["torch_cpu"])
            if "torch_cuda" in rng and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                except Exception:
                    pass

        return data

    def exists(self, filename: Union[str, Path]) -> bool:
        """Check whether a checkpoint file exists."""
        path = Path(filename)
        if not path.is_absolute():
            path = self.checkpoint_dir / path
        return path.is_file()

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint .pt files sorted by modification time."""
        if not self.checkpoint_dir.exists():
            return []
        files = [p for p in self.checkpoint_dir.glob("*.pt") if not p.name.endswith(".tmp")]
        files.sort(key=lambda p: p.stat().st_mtime)
        return files

    def latest_checkpoint(self) -> Optional[Path]:
        """Return path to the most recent checkpoint file, or None if empty."""
        files = self.list_checkpoints()
        return files[-1] if files else None
