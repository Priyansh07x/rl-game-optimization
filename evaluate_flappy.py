"""Formal Flappy Bird Tabular RL Evaluation Entry Point (Q-Learning & SARSA).

Evaluates trained tabular RL policies (Q-Learning & SARSA) in frozen mode:
- Exploration is disabled (eval_mode=True -> effective epsilon = 0.0).
- Pure greedy action selection is enforced.
- No Q-table updates or learning occurs.
- Exact environment, state discretization (75 states), actions, and rewards are preserved.
- Q-table integrity is verified before and after evaluation to guarantee immutability.
- Generates reproducible evaluation metrics CSV and exports clean learned Q-table models.
"""

import argparse
import copy
import csv
from pathlib import Path
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.agents.tabular.base_tabular import BaseTabularAgent
from src.agents.tabular.checkpoint import TabularCheckpointManager
from src.agents.tabular.q_learning import QLearningAgent
from src.agents.tabular.sarsa import SARSAAgent
from src.common.config import load_config
from src.common.logger import get_logger
from src.environments.flappy_env import FlappyEnv

# Explicit column schema for Flappy evaluation CSV
FLAPPY_EVALUATION_COLUMNS = [
    "episode",
    "score",
    "return",
    "reward",
    "episode_length",
    "success",
    "seed",
]


def parse_args():
    """Parse command-line arguments for Flappy Bird tabular RL evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Trained Flappy Bird Tabular RL Agent (Q-Learning / SARSA)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["q_learning", "sarsa"],
        help="Tabular RL algorithm ('q_learning' or 'sarsa'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained checkpoint or model .pkl file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for evaluation reproducibility (default: 42).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flappy_tabular.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluations/flappy",
        help="Directory to save evaluation CSV (default: evaluations/flappy).",
    )
    parser.add_argument(
        "--export-model",
        action="store_true",
        help="Export clean learned Q-table to models/flappy_<algo>_table.pkl.",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default=None,
        help="Explicit file path for exported clean model (overrides default).",
    )
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    """Set random seeds across Python and NumPy for evaluation reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def create_agent(
    algorithm: str,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> BaseTabularAgent:
    """Instantiate tabular agent (QLearningAgent or SARSAAgent).

    Args:
        algorithm: Algorithm identifier ('q_learning' or 'sarsa').
        config: Optional loaded configuration dictionary.
        seed: Optional random seed for agent exploration RNG.

    Returns:
        Configured BaseTabularAgent instance.
    """
    algo = algorithm.lower().strip()
    config = config or {}
    agent_cfg = config.get("agent", {})
    eps_cfg = config.get("epsilon", {})

    num_states = int(agent_cfg.get("num_states", 75))
    num_actions = int(agent_cfg.get("num_actions", 2))
    alpha = float(agent_cfg.get("alpha", 0.1))
    gamma = float(agent_cfg.get("gamma", 0.99))
    epsilon_start = float(eps_cfg.get("start", 1.0))
    epsilon_end = float(eps_cfg.get("end", 0.05))
    decay_episodes = int(eps_cfg.get("decay_episodes", 5000))

    if algo == "q_learning":
        return QLearningAgent(
            num_states=num_states,
            num_actions=num_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=decay_episodes,
            seed=seed,
        )
    elif algo == "sarsa":
        return SARSAAgent(
            num_states=num_states,
            num_actions=num_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=decay_episodes,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported algorithm '{algo}'. Expected 'q_learning' or 'sarsa'.")


def load_model_file(
    model_path: Union[str, Path],
    agent: BaseTabularAgent,
    expected_algorithm: Optional[str] = None,
) -> Dict[str, Any]:
    """Load Q-table weights from either a full training checkpoint or a clean model artifact.

    Args:
        model_path: Path to checkpoint or model pickle file.
        agent: BaseTabularAgent instance to load weights into.
        expected_algorithm: Optional algorithm name for strict validation.

    Returns:
        Loaded metadata dictionary.

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If file is malformed or incompatible.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model/checkpoint file not found at: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid model file format at {path}: top-level must be a dict.")

    # 1. Check if full checkpoint envelope from TabularCheckpointManager
    if "agent_state" in data:
        algo = data.get("algorithm", "").lower()
        if expected_algorithm and algo and algo != expected_algorithm.lower():
            raise ValueError(
                f"Algorithm mismatch in checkpoint at {path}: expected '{expected_algorithm}', got '{algo}'."
            )
        agent_state = data["agent_state"]
        agent.load_checkpoint_state(agent_state)
        return data

    # 2. Check if clean exported model artifact
    elif "q_table" in data:
        algo = data.get("algorithm", "").lower()
        if expected_algorithm and algo and algo != expected_algorithm.lower():
            raise ValueError(
                f"Algorithm mismatch in model file at {path}: expected '{expected_algorithm}', got '{algo}'."
            )
        q_table = np.array(data["q_table"], dtype=np.float64)
        if q_table.shape != (agent.num_states, agent.num_actions):
            raise ValueError(
                f"Q-table shape {q_table.shape} does not match expected agent shape ({agent.num_states}, {agent.num_actions})."
            )
        agent.q_table = q_table
        return data

    else:
        raise ValueError(f"Unrecognized tabular model format in {path}: neither 'agent_state' nor 'q_table' found.")


def export_clean_model(
    agent: BaseTabularAgent,
    algorithm: str,
    export_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    evaluation_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export clean tabular Q-table artifact without training or exploration state.

    Args:
        agent: Evaluated BaseTabularAgent containing final Q-table.
        algorithm: Algorithm identifier string ('q_learning' or 'sarsa').
        export_path: Destination path for exported model pickle file.
        config: Optional configuration dictionary.
        evaluation_summary: Optional summary stats dictionary.

    Returns:
        Path to the exported model file.
    """
    target_path = Path(export_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".pkl.tmp")

    model_data: Dict[str, Any] = {
        "format_version": "1.0",
        "model_type": "tabular_q_table",
        "algorithm": str(algorithm).lower(),
        "q_table": np.array(agent.q_table, dtype=np.float64, copy=True),
        "num_states": int(agent.num_states),
        "num_actions": int(agent.num_actions),
        "action_mapping": {0: "FLAP", 1: "NO_FLAP"},
        "state_discretization": {
            "num_dx_bins": 5,
            "num_dy_bins": 5,
            "num_vy_bins": 3,
            "total_states": 75,
            "dx_edges": [0.0, 0.15, 0.30, 0.50],
            "dy_edges": [-0.10, -0.02, 0.02, 0.10],
            "vy_edges": [-0.20, 0.20],
        },
        "evaluation_summary": copy.deepcopy(evaluation_summary) if evaluation_summary is not None else {},
    }

    with open(temp_path, "wb") as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    temp_path.replace(target_path)
    return target_path


def run_evaluation(
    algorithm: Optional[str] = None,
    model_path: Optional[Union[str, Path]] = None,
    episodes: int = 100,
    seed: int = 42,
    config_path: str = "configs/flappy_tabular.yaml",
    output_dir: str = "evaluations/flappy",
    export_model: bool = False,
    export_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute formal evaluation of trained Flappy Bird tabular agent.

    Args:
        algorithm: Optional algorithm ('q_learning' or 'sarsa'). Auto-detected if None.
        model_path: Optional path to checkpoint file. Defaults to best checkpoint if None.
        episodes: Number of evaluation episodes (default: 100).
        seed: Random seed for deterministic evaluation.
        config_path: Path to YAML config file.
        output_dir: Directory to save evaluation CSV metrics.
        export_model: Whether to export clean Q-table model artifact to models/.
        export_path: Optional explicit path for exported model.

    Returns:
        Summary dictionary with evaluation statistics.

    Raises:
        ValueError: If configuration, algorithm, or model path is invalid.
        RuntimeError: If Q-table immutability integrity check fails.
    """
    setup_seed(seed)

    # 1. Load config if available
    config = {}
    if Path(config_path).is_file():
        config = load_config(config_path)

    # 2. Resolve Algorithm and Default Model Path
    resolved_algo = algorithm
    resolved_model_path = model_path

    if resolved_model_path is None:
        if resolved_algo is None:
            resolved_algo = config.get("agent", {}).get("algorithm", "q_learning")
        resolved_algo = resolved_algo.lower().strip()
        # Default to best checkpoint produced by Part 5
        candidate_paths = [
            f"checkpoints/flappy/flappy_{resolved_algo}_best.pkl",
            f"checkpoints/flappy/{resolved_algo}/flappy_{resolved_algo}_best.pkl",
            f"checkpoints/flappy/checkpoint_ep5000.pkl",
            f"models/flappy_{resolved_algo}_table.pkl",
        ]
        for candidate in candidate_paths:
            if Path(candidate).is_file():
                resolved_model_path = candidate
                break
        if resolved_model_path is None:
            raise FileNotFoundError(
                f"No default model checkpoint found for '{resolved_algo}'. Please specify --model explicitly."
            )
    else:
        resolved_model_path = str(resolved_model_path)
        if resolved_algo is None:
            # Try auto-detecting algorithm from filename or checkpoint envelope
            if "sarsa" in resolved_model_path.lower():
                resolved_algo = "sarsa"
            elif "q_learning" in resolved_model_path.lower() or "q_table" in resolved_model_path.lower():
                resolved_algo = "q_learning"
            else:
                # Peek checkpoint file
                try:
                    with open(resolved_model_path, "rb") as f:
                        peek = pickle.load(f)
                    if isinstance(peek, dict) and "algorithm" in peek:
                        resolved_algo = str(peek["algorithm"]).lower()
                except Exception:
                    resolved_algo = "q_learning"

    resolved_algo = (resolved_algo or "q_learning").lower().strip()
    if resolved_algo not in ("q_learning", "sarsa"):
        raise ValueError(f"Invalid algorithm '{resolved_algo}'. Expected 'q_learning' or 'sarsa'.")

    # 3. Setup Directories & Logger
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    logger = get_logger(name=f"flappy_{resolved_algo}_evaluation", log_dir=out_dir_path)
    logger.info(
        f"Starting Flappy Evaluation | Algorithm: {resolved_algo} | Model: {resolved_model_path} | "
        f"Episodes: {episodes} | Seed: {seed}"
    )

    # 4. Instantiate Agent & Load Model Weights
    agent = create_agent(algorithm=resolved_algo, config=config, seed=seed)
    loaded_metadata = load_model_file(
        model_path=resolved_model_path,
        agent=agent,
        expected_algorithm=resolved_algo,
    )
    logger.info(
        f"Successfully loaded {resolved_algo} model from {resolved_model_path} "
        f"({agent.non_zero_entries}/{agent.q_table_size} non-zero Q-values)"
    )

    # 5. Capture Integrity Snapshot of Q-Table before Evaluation
    q_table_snapshot = np.copy(agent.q_table)

    # 6. Initialize Flappy Environment
    env = FlappyEnv(
        survival_reward=0.5,
        pipe_pass_reward=5.0,
        collision_reward=-1000.0,
    )

    # 7. Run Evaluation Loop with Deterministic Seeding (seed + episode_number)
    records: List[Dict[str, Any]] = []

    for ep in range(1, episodes + 1):
        ep_seed = seed + ep
        raw_obs, info = env.reset(seed=ep_seed)
        state = info["state_index"]

        ep_reward = 0.0
        ep_length = 0
        done = False

        while not done:
            # Strictly greedy action selection with zero exploration (eval_mode=True)
            action = agent.select_action(state, eval_mode=True)

            next_raw, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_length += 1
            done = terminated or truncated
            state = info["state_index"]

        score = float(info.get("score", 0))
        success = 1 if score > 0 else 0

        record = {
            "episode": ep,
            "score": score,
            "return": ep_reward,
            "reward": ep_reward,
            "episode_length": ep_length,
            "success": success,
            "seed": seed,
        }
        records.append(record)

    env.close()

    # 8. Verify Q-Table Integrity Post-Evaluation (Must be 100% untouched)
    is_q_table_intact = np.array_equal(q_table_snapshot, agent.q_table)
    if not is_q_table_intact:
        raise RuntimeError(
            "CRITICAL INTEGRITY FAILURE: Q-table was modified during evaluation! "
            "Evaluation must be strictly frozen."
        )
    logger.info("Q-table immutability integrity check PASSED: Q-table remained completely unmodified.")

    # 9. Save Evaluation Metrics CSV (evaluations/flappy/<algo>.csv)
    csv_filename = f"{resolved_algo}.csv"
    metrics_csv_path = out_dir_path / csv_filename
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FLAPPY_EVALUATION_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Evaluation metrics successfully saved to: {metrics_csv_path}")

    # 10. Compute Summary Statistics
    returns = [r["return"] for r in records]
    scores = [r["score"] for r in records]
    lengths = [r["episode_length"] for r in records]
    success_count = sum(r["success"] for r in records)
    success_rate = (success_count / episodes) * 100.0 if episodes > 0 else 0.0

    stats: Dict[str, Any] = {
        "algorithm": resolved_algo,
        "model_path": str(resolved_model_path),
        "episodes": episodes,
        "seed": seed,
        "epsilon": 0.0,
        "mean_return": float(np.mean(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        "mean_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "success_count": int(success_count),
        "success_rate": float(success_rate),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "q_table_integrity_passed": is_q_table_intact,
        "metrics_csv_path": str(metrics_csv_path),
        "exported_model_path": None,
    }

    # 11. Export Clean Learned Q-Table Model if requested
    if export_model:
        if export_path is not None:
            resolved_export_path = Path(export_path)
        else:
            # Official required path: models/flappy_q_table.pkl or models/flappy_sarsa_table.pkl
            table_name = "flappy_q_table.pkl" if resolved_algo == "q_learning" else "flappy_sarsa_table.pkl"
            resolved_export_path = Path("models") / table_name

        exported_file = export_clean_model(
            agent=agent,
            algorithm=resolved_algo,
            export_path=resolved_export_path,
            config=config,
            evaluation_summary=stats,
        )
        stats["exported_model_path"] = str(exported_file)
        logger.info(f"Clean model artifact exported to: {exported_file}")

    return stats


def main():
    """CLI entry point for Flappy Bird evaluation."""
    args = parse_args()
    stats = run_evaluation(
        algorithm=args.algorithm,
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        config_path=args.config,
        output_dir=args.output_dir,
        export_model=args.export_model,
        export_path=args.export_path,
    )

    print("\n========================================================")
    print(f"   Flappy Bird Evaluation Complete: {stats['algorithm'].upper()}")
    print("========================================================")
    print(f"Model Path:         {stats['model_path']}")
    print(f"Episodes:           {stats['episodes']}")
    print(f"Seed:               {stats['seed']}")
    print(f"Exploration:        Frozen (epsilon = {stats['epsilon']})")
    print(f"Mean Return:        {stats['mean_return']:.2f} (Min: {stats['min_return']:.1f}, Max: {stats['max_return']:.1f}, Median: {stats['median_return']:.1f})")
    print(f"Mean Score:         {stats['mean_score']:.3f} (Max: {int(stats['max_score'])})")
    print(f"Success Count:      {stats['success_count']}/{stats['episodes']} ({stats['success_rate']:.1f}% scored >= 1 pipe)")
    print(f"Mean Steps:         {stats['mean_length']:.1f} (Min: {stats['min_length']}, Max: {stats['max_length']})")
    print(f"Integrity Check:    {'PASSED' if stats['q_table_integrity_passed'] else 'FAILED'}")
    print(f"Evaluation CSV:     {stats['metrics_csv_path']}")
    if stats["exported_model_path"]:
        print(f"Exported Model:     {stats['exported_model_path']}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
