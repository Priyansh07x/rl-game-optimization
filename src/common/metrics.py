"""Structured CSV metrics logger for reinforcement learning training runs."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


DEFAULT_METRICS_COLUMNS: List[str] = [
    "episode",
    "reward",
    "episode_length",
    "loss",
    "epsilon",
    "avg_q",
    "score",
    "seed",
]


class MetricsLogger:
    """CSV metrics logger for recording episode-level training metrics."""

    def __init__(
        self,
        log_path: Union[str, Path] = "logs/snake/training_metrics.csv",
        columns: Optional[List[str]] = None,
        append: bool = True,
    ):
        """Initialize MetricsLogger.

        Args:
            log_path: Path to the CSV metrics output file (default: 'logs/snake/training_metrics.csv').
            columns: List of CSV column names (default: standard 8-column RL metrics schema).
            append: If True, append to existing file without re-writing header.
        """
        self.log_path = Path(log_path)
        self.columns = list(columns) if columns is not None else DEFAULT_METRICS_COLUMNS
        self.append = append

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._open_file()

    def _open_file(self) -> None:
        """Open CSV file in append/write mode and handle header writing."""
        file_exists = self.log_path.exists() and self.log_path.stat().st_size > 0

        mode = "a" if (file_exists and self.append) else "w"
        self._file = open(self.log_path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns, extrasaction="ignore")

        if not file_exists or not self.append or mode == "w":
            self._writer.writeheader()
            self._file.flush()

    def log_episode(
        self,
        episode: int,
        reward: float,
        episode_length: int,
        loss: Optional[float],
        epsilon: float,
        avg_q: Optional[float],
        score: float,
        seed: int,
        **extra_metrics: Any,
    ) -> None:
        """Log a single episode's metrics row.

        Args:
            episode: Episode integer index.
            reward: Cumulative episode return / reward.
            episode_length: Number of environment steps in episode.
            loss: Mean learning loss float, or None if no learning updates occurred.
            epsilon: Current exploration rate epsilon.
            avg_q: Mean Q-value float, or None if no learning updates occurred.
            score: Native game score achieved.
            seed: Random seed used.
            **extra_metrics: Optional additional custom column key-value pairs.
        """
        if self._file is None or self._writer is None or self._file.closed:
            self._open_file()

        row: Dict[str, Any] = {
            "episode": int(episode),
            "reward": float(reward),
            "episode_length": int(episode_length),
            "loss": "" if loss is None else float(loss),
            "epsilon": float(epsilon),
            "avg_q": "" if avg_q is None else float(avg_q),
            "score": float(score),
            "seed": int(seed),
        }
        for k, v in extra_metrics.items():
            if k in self.columns:
                row[k] = "" if v is None else v

        self._writer.writerow(row)
        self.flush()

    def flush(self) -> None:
        """Flush unwritten CSV buffers to disk."""
        if self._file is not None and not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
