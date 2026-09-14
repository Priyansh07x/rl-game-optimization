"""Unit tests for MetricsLogger implementation."""

import csv
from pathlib import Path
import pytest

from src.common.metrics import DEFAULT_METRICS_COLUMNS, MetricsLogger


class TestMetricsLogger:
    """Test suite for MetricsLogger CSV generation, append logic, missing values, and file durability."""

    def test_logger_initialization_and_directory_creation(self, tmp_path: Path):
        """1 & 2. Verify logger initialization and output directory creation."""
        csv_path = tmp_path / "logs" / "snake" / "training_metrics.csv"
        assert not csv_path.parent.exists()

        logger = MetricsLogger(log_path=csv_path)
        assert logger.log_path == csv_path
        assert csv_path.parent.is_dir()
        logger.close()

    def test_csv_file_and_header_creation(self, tmp_path: Path):
        """3 & 4. Verify CSV file is created with the exact expected header."""
        csv_path = tmp_path / "metrics.csv"
        logger = MetricsLogger(log_path=csv_path)
        logger.close()

        assert csv_path.is_file()
        content = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(content) == 1
        header = content[0].split(",")
        assert header == DEFAULT_METRICS_COLUMNS
        assert header == ["episode", "reward", "episode_length", "loss", "epsilon", "avg_q", "score", "seed"]

    def test_write_single_episode_row(self, tmp_path: Path):
        """5 & 6. Verify writing one episode row writes correct values and column order."""
        csv_path = tmp_path / "metrics.csv"
        logger = MetricsLogger(log_path=csv_path)
        logger.log_episode(
            episode=1,
            reward=2.5,
            episode_length=15,
            loss=0.045,
            epsilon=0.98,
            avg_q=1.23,
            score=2.0,
            seed=42,
        )
        logger.close()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        assert len(reader) == 1
        row = reader[0]
        assert row["episode"] == "1"
        assert float(row["reward"]) == 2.5
        assert row["episode_length"] == "15"
        assert float(row["loss"]) == 0.045
        assert float(row["epsilon"]) == 0.98
        assert float(row["avg_q"]) == 1.23
        assert float(row["score"]) == 2.0
        assert row["seed"] == "42"

    def test_multiple_episode_rows(self, tmp_path: Path):
        """7. Verify writing multiple episode rows."""
        csv_path = tmp_path / "metrics.csv"
        with MetricsLogger(log_path=csv_path) as logger:
            for ep in range(1, 6):
                logger.log_episode(
                    episode=ep,
                    reward=float(ep * 1.5),
                    episode_length=10 + ep,
                    loss=0.1 / ep,
                    epsilon=1.0 - ep * 0.1,
                    avg_q=0.5 * ep,
                    score=float(ep),
                    seed=42,
                )

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()

        assert len(lines) == 6  # Header + 5 rows

    def test_append_resume_behavior_and_header_deduplication(self, tmp_path: Path):
        """8 & 9. Verify append mode adds rows without duplicating header."""
        csv_path = tmp_path / "metrics.csv"

        # Session 1: Write episodes 1 and 2
        logger1 = MetricsLogger(log_path=csv_path, append=True)
        logger1.log_episode(1, 1.0, 10, None, 1.0, None, 1.0, 42)
        logger1.log_episode(2, 2.0, 20, 0.05, 0.9, 0.5, 2.0, 42)
        logger1.close()

        # Session 2: Reopen in append mode and write episode 3
        logger2 = MetricsLogger(log_path=csv_path, append=True)
        logger2.log_episode(3, 3.0, 30, 0.04, 0.8, 0.8, 3.0, 42)
        logger2.close()

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()

        # Header (1) + 3 data rows = 4 lines total
        assert len(lines) == 4
        assert lines[0] == "episode,reward,episode_length,loss,epsilon,avg_q,score,seed"

        # Count header occurrences
        header_count = sum(1 for line in lines if line == lines[0])
        assert header_count == 1, "Header must not be duplicated when appending"

    def test_missing_optional_metrics(self, tmp_path: Path):
        """10. Verify missing loss and avg_q values are logged as empty fields."""
        csv_path = tmp_path / "metrics.csv"
        logger = MetricsLogger(log_path=csv_path)
        logger.log_episode(
            episode=1,
            reward=0.0,
            episode_length=5,
            loss=None,  # Unavailable
            epsilon=1.0,
            avg_q=None,  # Unavailable
            score=0.0,
            seed=42,
        )
        logger.close()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        row = reader[0]
        assert row["loss"] == ""
        assert row["avg_q"] == ""
        assert row["episode"] == "1"

    def test_flush_and_close_behavior(self, tmp_path: Path):
        """11. Verify explicit flush and close write content to disk."""
        csv_path = tmp_path / "metrics.csv"
        logger = MetricsLogger(log_path=csv_path)
        logger.log_episode(1, 1.0, 10, 0.1, 0.9, 1.0, 1.0, 42)
        logger.flush()

        # Check content is on disk after flush
        assert csv_path.stat().st_size > 0

        logger.close()
        assert logger._file is None

    def test_existing_data_not_overwritten(self, tmp_path: Path):
        """12. Verify existing CSV data is preserved when append=True."""
        csv_path = tmp_path / "metrics.csv"

        # Pre-create file with existing content
        csv_path.write_text(
            "episode,reward,episode_length,loss,epsilon,avg_q,score,seed\n1,10.0,10,0.1,1.0,0.5,1.0,42\n",
            encoding="utf-8",
        )

        logger = MetricsLogger(log_path=csv_path, append=True)
        logger.log_episode(2, 20.0, 20, 0.08, 0.9, 1.0, 2.0, 42)
        logger.close()

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["episode"] == "1"
        assert rows[1]["episode"] == "2"
