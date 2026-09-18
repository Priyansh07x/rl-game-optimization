"""Tabular Reinforcement Learning Checkpoint Manager.

Handles atomic serialization, validation, and restoration of tabular RL agents
(Q-Learning and SARSA) using Python pickle format.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import copy
import pickle
import numpy as np

from src.agents.tabular.base_tabular import BaseTabularAgent


class TabularCheckpointManager:
    """Manages saving, loading, listing, and validation of tabular RL agent checkpoints."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "checkpoints/flappy",
        format_version: str = "1.0",
    ):
        """Initialize TabularCheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoint .pkl files are stored.
            format_version: Version identifier string for saved checkpoint envelopes.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.format_version = str(format_version)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        agent: BaseTabularAgent,
        episode: int,
        algorithm: str,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        filename: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Atomically save tabular agent state and envelope metadata to a pickle file.

        Args:
            agent: BaseTabularAgent instance (QLearningAgent or SARSAAgent).
            episode: Current training episode count.
            algorithm: Algorithm identifier string ('q_learning' or 'sarsa').
            config: Optional configuration dictionary.
            seed: Optional random seed used for training.
            filename: Optional target filename (defaults to 'checkpoint_ep<episode>.pkl').
            extra_info: Optional arbitrary extra metadata dictionary.

        Returns:
            Path object pointing to the created checkpoint file.

        Raises:
            TypeError: If agent is not an instance of BaseTabularAgent.
        """
        if not isinstance(agent, BaseTabularAgent):
            raise TypeError(f"agent must be an instance of BaseTabularAgent, got {type(agent)}")

        if filename is None:
            filename = f"checkpoint_ep{episode}.pkl"

        target_path = self.checkpoint_dir / filename
        temp_path = target_path.with_suffix(".pkl.tmp")

        checkpoint_data: Dict[str, Any] = {
            "format_version": self.format_version,
            "algorithm": str(algorithm),
            "episode": int(episode),
            "epsilon": float(agent.epsilon),
            "agent_state": agent.get_checkpoint_state(),
            "config": copy.deepcopy(config) if config is not None else None,
            "seed": seed,
            "extra_info": copy.deepcopy(extra_info) if extra_info is not None else {},
        }

        # Atomic write: serialize to temp file first, then atomically replace destination
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        temp_path.replace(target_path)
        return target_path

    def load(
        self,
        checkpoint_path: Union[str, Path],
        agent: Optional[BaseTabularAgent] = None,
        restore_rng: bool = True,
        expected_algorithm: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load and validate checkpoint file and optionally restore state into an agent.

        Args:
            checkpoint_path: Path to checkpoint file (absolute or relative to checkpoint_dir).
            agent: Optional BaseTabularAgent instance into which state will be restored.
            restore_rng: If True and agent provided, restores the agent's RNG state.
                         If False, preserves the agent's current RNG state.
            expected_algorithm: Optional algorithm name to strictly validate against ('q_learning'/'sarsa').

        Returns:
            Dictionary containing complete checkpoint envelope and state data.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            ValueError: If checkpoint format, envelope, or Q-table is malformed/incompatible.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            # Try relative to checkpoint_dir
            alt_path = self.checkpoint_dir / path
            if alt_path.is_file():
                path = alt_path
            else:
                raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint file at {path}: {e}")

        if not isinstance(data, dict):
            raise ValueError(f"Invalid checkpoint format at {path}: top-level must be a dictionary.")

        # Validate required envelope fields
        required_fields = ["format_version", "algorithm", "episode", "epsilon", "agent_state"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Invalid checkpoint format at {path}: missing required field '{field}'.")

        if data["format_version"] != self.format_version:
            raise ValueError(
                f"Incompatible checkpoint format version '{data['format_version']}', expected '{self.format_version}'."
            )

        if expected_algorithm is not None and data["algorithm"] != expected_algorithm:
            raise ValueError(
                f"Algorithm mismatch in checkpoint: expected '{expected_algorithm}', got '{data['algorithm']}'."
            )

        agent_state = data.get("agent_state")
        if not isinstance(agent_state, dict) or "q_table" not in agent_state:
            raise ValueError(f"Invalid agent_state in checkpoint at {path}: missing 'q_table'.")

        loaded_q_table = np.asarray(agent_state["q_table"])

        if agent is not None:
            if not isinstance(agent, BaseTabularAgent):
                raise TypeError(f"agent must be an instance of BaseTabularAgent, got {type(agent)}")

            expected_shape = (agent.num_states, agent.num_actions)
            if loaded_q_table.shape != expected_shape:
                raise ValueError(
                    f"Incompatible Q-table shape {loaded_q_table.shape} in checkpoint, expected {expected_shape}."
                )

            # Preserve RNG state if restore_rng is False
            saved_rng_state = agent.rng.bit_generator.state if not restore_rng else None

            # Deep copy state dict so loaded agent state is fully independent
            agent.load_checkpoint_state(copy.deepcopy(agent_state))

            if not restore_rng and saved_rng_state is not None:
                agent.rng.bit_generator.state = saved_rng_state

        return data

    def exists(self, filename: Union[str, Path]) -> bool:
        """Check whether a checkpoint file exists in checkpoint_dir or at path.

        Args:
            filename: Filename or path to verify.

        Returns:
            True if file exists, False otherwise.
        """
        path = Path(filename)
        if not path.is_absolute():
            path = self.checkpoint_dir / path
        return path.is_file()

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint .pkl files sorted by episode index.

        Returns:
            Sorted list of Path objects.
        """
        if not self.checkpoint_dir.exists():
            return []

        files = [p for p in self.checkpoint_dir.glob("*.pkl") if not p.name.endswith(".tmp")]

        def _sort_key(p: Path) -> int:
            stem = p.stem
            if stem.startswith("checkpoint_ep"):
                try:
                    return int(stem.replace("checkpoint_ep", ""))
                except ValueError:
                    pass
            return int(p.stat().st_mtime)

        files.sort(key=_sort_key)
        return files

    def latest_checkpoint(self) -> Optional[Path]:
        """Return the path to the newest checkpoint file, or None if no checkpoints exist.

        Returns:
            Path object or None.
        """
        files = self.list_checkpoints()
        return files[-1] if files else None
