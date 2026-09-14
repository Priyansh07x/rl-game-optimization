# Reinforcement Learning for Game Optimization

A reinforcement learning project that develops game-playing agents for **Snake–Apple, Flappy Bird, and Pac-Man** using tabular and deep reinforcement learning techniques.

> **Current status:** Parts 0–3 are completed. The current implementation includes the complete **Snake environment and Snake DQN pipeline**. Flappy Bird and Pac-Man are planned for subsequent parts.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Objective](#project-objective)
- [Research Question](#research-question)
- [Project Status](#project-status)
- [Planned Algorithm Matrix](#planned-algorithm-matrix)
- [System Architecture](#system-architecture)
- [Current Snake Implementation](#current-snake-implementation)
- [Snake Environment](#snake-environment)
- [Snake State Representation](#snake-state-representation)
- [Snake Actions](#snake-actions)
- [Snake Reward](#snake-reward)
- [Snake DQN Architecture](#snake-dqn-architecture)
- [Dual Replay Strategy](#dual-replay-strategy)
- [Training Configuration](#training-configuration)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Tests](#running-the-tests)
- [Training Snake DQN](#training-snake-dqn)
- [Evaluating Snake DQN](#evaluating-snake-dqn)
- [Checkpointing and Resuming](#checkpointing-and-resuming)
- [Generated Artifacts](#generated-artifacts)
- [Reproducibility](#reproducibility)
- [Development Workflow](#development-workflow)
- [Future Roadmap](#future-roadmap)
- [Evaluation Methodology](#evaluation-methodology)
- [Research Basis](#research-basis)
- [Limitations](#limitations)
- [Project Deliverables](#project-deliverables)
- [License](#license)

---

# Project Overview

This project investigates how reinforcement learning can be used to optimize game-playing behavior across environments with different state representations and levels of complexity.

The complete project is designed around three games:

| Game | Primary Algorithm | State Representation | Learning Type |
|---|---|---|---|
| Snake–Apple | Refined DQN | `(4, 64, 64)` | Deep, off-policy |
| Flappy Bird | Q-Learning + SARSA | 75 discrete states | Tabular |
| Pac-Man | DQN | `(4, 84, 84)` | Deep, off-policy |

The project is being developed incrementally so that the environment, state representation, learning algorithm, training infrastructure, and evaluation methodology can each be verified independently.

The current implementation has reached **Part 3: Snake DQN**.

---

# Project Objective

The primary objective is to implement and evaluate reinforcement learning agents capable of learning game-playing strategies through interaction with their environments.

The project focuses on:

- Reinforcement learning environment design
- Markov Decision Process formulation
- State representation
- Reward engineering
- Epsilon-greedy exploration
- Q-Learning
- SARSA
- Deep Q-Networks
- Dueling DQN
- Experience replay
- Target networks
- Model checkpointing
- Training logging
- Reproducible evaluation
- Baseline comparison
- Failure analysis

The project is **not** based on a hard-coded sequence of optimal moves. The agents are intended to learn behavior from game interaction.

---

# Research Question

The central research question is:

> **How do state representation, reward design, exploration strategy, and reinforcement learning algorithm affect learning and performance across different games?**

The completed and planned experiments are designed to investigate this question through controlled environments and measurable evaluation.

---

# Project Status

## Completed

### Part 0 — Project Setup

- Python development environment
- Virtual environment support
- Git repository structure
- Dependency management
- Core project configuration

### Part 1 — Common RL Infrastructure

- Common project organization
- Environment/agent separation
- Gymnasium-based environment conventions
- Testing infrastructure
- Configuration management
- Training/evaluation separation

### Part 2 — Snake Environment

- Snake–Apple environment
- Reset behavior
- Action handling
- Movement
- Apple spawning
- Snake growth
- Wall collision
- Self collision
- Reward handling
- Observation generation
- Environment validation
- Automated tests

### Part 3 — Snake DQN

- Snake DQN agent
- CNN-based feature extraction
- Dueling DQN architecture
- Experience replay
- Dual replay buffers
- Target network
- Epsilon-greedy exploration
- Training loop
- Logging
- Checkpointing
- Checkpoint resuming
- Model evaluation

## Current milestone

```text
PART 0
Project Setup
    |
    v
PART 1
Common RL Infrastructure
    |
    v
PART 2
Build + Verify Snake
    |
    v
PART 3
Train Snake DQN
    |
    v
CURRENT STATUS
```

## Not yet completed

```text
PART 4  -> Build + verify Flappy Bird
PART 5  -> Train Flappy Q-Learning + SARSA
PART 6  -> Build + verify Pac-Man
PART 7  -> Train Pac-Man DQN
PART 8  -> Unified evaluation
PART 9  -> Ablation + failure analysis
PART 10 -> Final plots + report + demonstration
```

---

# Planned Algorithm Matrix

The final project is intended to use the following algorithm/game combinations:

```text
                    GAME ENVIRONMENTS
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
        Snake           Flappy Bird       Pac-Man
          |                |                |
          v                v                v
       Dueling         Q-Learning +         DQN
         DQN              SARSA
          |                |                |
          +----------------+----------------+
                           |
                           v
                    Unified Evaluation
                           |
                           v
              Comparison + Ablation Study
```

The current repository should not be interpreted as containing the complete three-game implementation yet.

---

# System Architecture

The project follows a separation between the game environment and the learning agent.

```text
+---------------------------+
|      GAME ENVIRONMENT     |
|                           |
|  Snake / Flappy / Pac-Man |
+-------------+-------------+
              |
              | observation
              v
+---------------------------+
|     STATE PROCESSING      |
|                           |
| preprocessing / encoding  |
+-------------+-------------+
              |
              | state
              v
+---------------------------+
|        RL AGENT           |
|                           |
| Q-Learning / SARSA / DQN  |
+-------------+-------------+
              |
              | action
              v
+---------------------------+
|       GAME ENGINE         |
+-------------+-------------+
              |
        +-----+-----+
        |           |
        v           v
     reward      next state
        |           |
        +-----+-----+
              |
              v
       LEARNING UPDATE
              |
              +----------> repeat
```

The environment does not contain the neural network or learning logic.

This separation allows the environment to be tested independently from the agent.

---

# Current Snake Implementation

The current completed implementation focuses on a Snake–Apple environment and a refined DQN agent.

The Snake implementation is based on the project milestone specifications and the supplied Snake reinforcement learning research.

---

# Snake Environment

The Snake environment is based on the following specification:

- Game resolution: `240 × 240`
- Logical grid: `12 × 12`
- Initial snake length: `3`
- Initial direction: `RIGHT`
- Random initial snake/apple deployment
- One apple at a time
- Snake grows after eating an apple
- A new apple is spawned after an apple is eaten
- Four movement actions
- Episode termination on wall collision
- Episode termination on self collision

The conceptual environment loop is:

```text
RESET
  |
  v
OBSERVE
  |
  v
SELECT ACTION
  |
  v
STEP
  |
  +------> REWARD
  |
  +------> NEXT STATE
  |
  +------> TERMINATION
  |
  v
REPEAT
```

---

# Snake State Representation

The Snake visual state is processed through multiple stages:

```text
240 × 240 Game
      |
      v
12 × 12 Logical Grid
      |
      v
Rendered Frame
      |
      v
RGB Image
      |
      v
Resize to 64 × 64
      |
      v
HSV / Background Processing
      |
      v
Processed Frame
      |
      v
4-Frame History
      |
      v
DQN Input
```

The official DQN input convention is:

```text
(4, 64, 64)
```

This represents four processed frames forming a short temporal history.

---

# Snake Actions

The action mapping is:

```text
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

The mapping is kept explicit so that numeric action values do not become scattered unexplained constants throughout the implementation.

---

# Snake Reward

The underlying Snake research describes the basic reward structure as:

```text
Apple eaten  -> +1
Collision     -> -1
```

The refined Milestone 3 implementation uses:

```text
Apple eaten  -> +1.0
Survival     -> +0.1
Collision    -> -1.0
```

Therefore, the implementation distinguishes between the research-paper reward description and the final project training configuration.

The project configuration is the authoritative source for the actual experiment.

---

# Snake DQN Architecture

The current Snake agent uses a **Dueling DQN** architecture.

Conceptually:

```text
              INPUT
          (4, 64, 64)
                |
                v
        +---------------+
        | CNN Layer     |
        | 16 filters    |
        +---------------+
                |
                v
        +---------------+
        | CNN Layer     |
        | 32 filters    |
        +---------------+
                |
                v
        +---------------+
        | CNN Layer     |
        | 64 filters    |
        +---------------+
                |
                v
        Shared Features
           /       \
          /         \
         v           v
   Value Stream   Advantage Stream
        |               |
        v               v
       V(s)            A(s,a)
          \             /
           \           /
            +---------+
                 |
                 v
              Q(s,a)
```

The network contains:

- Three convolutional stages
- A fully connected layer with 256 units
- A dueling value/advantage head

---

# Dual Replay Strategy

The refined Snake DQN uses two experience buffers based on temporal-difference error.

```text
                    EXPERIENCE
                         |
                  TD-error calculation
                         |
              +----------+----------+
              |                     |
              v                     v
       HIGH TD ERROR          LOW TD ERROR
          BUFFER                 BUFFER
        35,000 entries         15,000 entries
              \                     /
               \                   /
                +--------+--------+
                         |
                         v
                   SAMPLE BATCH
                         |
                         v
                     DQN UPDATE
```

Configuration:

```text
Total replay capacity = 50,000

High-error capacity   = 35,000
Low-error capacity    = 15,000

TD threshold          = 0.5

High-error sampling   = 70%
Low-error sampling    = 30%
```

The purpose of the dual replay strategy is to increase the influence of informative transitions while retaining lower-error experiences.

---

# Training Configuration

The official Milestone 3 Snake DQN configuration is:

| Parameter | Value |
|---|---:|
| Input shape | `(4, 64, 64)` |
| CNN filters | 16, 32, 64 |
| Fully connected layer | 256 |
| Network head | Dueling DQN |
| Replay capacity | 50,000 |
| High-error buffer | 35,000 |
| Low-error buffer | 15,000 |
| TD threshold | 0.5 |
| High-error sampling | 70% |
| Low-error sampling | 30% |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Gamma | 0.99 |
| Target update | Every 500 steps |
| Training gap | 4 frames |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.02 |
| Epsilon decay | 200,000 frames |
| Checkpoint frequency | Every 100 episodes |

---

# Repository Structure

The repository is organized so that game environments, agents, configurations, experiments, logs, models, and evaluation artifacts remain separated.

```text
Cornerstone-project/
│
├── checkpoints/
│   └── snake/
│
├── configs/
│   └── snake_dqn.yaml
│
├── evaluations/
│
├── games/
│
├── logs/
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── agents/
│   │   ├── dqn/
│   │   └── tabular/
│   │
│   ├── environments/
│   │
│   ├── evaluation/
│   │
│   ├── preprocessing/
│   │
│   └── training/
│
├── tests/
│
├── train_snake.py
├── evaluate_snake.py
├── requirements.txt
├── .gitignore
└── README.md
```

The exact contents of some directories will expand as Flappy Bird and Pac-Man are implemented.

---

# Requirements

## Python

Current development version:

```text
Python 3.14.0
```

## Core dependencies

The current project dependency specification is:

```text
gymnasium>=1.0.0
numpy>=2.0.0
pytest>=8.0.0
pyyaml>=6.0.0
torch>=2.0.0
```

These dependencies are listed in `requirements.txt`.

Additional dependencies may be introduced later when the Flappy Bird and Pac-Man stages are implemented.

---

# Installation

Clone the repository and create a virtual environment.

## Windows PowerShell

```powershell
git clone <repository-url>
cd Cornerstone-project

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify Python:

```powershell
python --version
```

Expected current development version:

```text
Python 3.14.0
```

The project intentionally uses a virtual environment so that project dependencies remain isolated from the system Python installation.

---

# Running the Tests

The complete test suite can be executed with:

```powershell
python -m pytest tests/
```

For verbose output:

```powershell
pytest tests/ -v
```

The test suite is used to verify the implemented project components before relying on training results.

---

# Training Snake DQN

The default Snake training command is:

```powershell
python train_snake.py
```

A custom configuration, episode count, and seed can be specified:

```powershell
python train_snake.py --config configs/snake_dqn.yaml --episodes 1000 --seed 42
```

The training process includes:

```text
Environment
    |
    v
State preprocessing
    |
    v
Frame stacking
    |
    v
Epsilon-greedy action selection
    |
    v
Environment interaction
    |
    v
Transition
    |
    v
Dual replay buffers
    |
    v
DQN update
    |
    v
Target network update
    |
    v
Logging + checkpointing
```

---

# Evaluating Snake DQN

Evaluation is performed separately from training.

Use:

```powershell
python evaluate_snake.py --model models/snake_dqn_best.pt --episodes 100
```

To evaluate a specific checkpoint:

```powershell
python evaluate_snake.py --model checkpoints/snake/checkpoint_ep1000.pt --episodes 100
```

Evaluation should use a frozen model and should not continue learning.

The standard project evaluation target is **100 episodes**, allowing metrics such as mean and standard deviation to be reported rather than relying on a single lucky episode.

---

# Checkpointing and Resuming

Snake training supports checkpoint resuming.

Example:

```powershell
python train_snake.py --resume checkpoints/snake/checkpoint_ep500.pt
```

Conceptually:

```text
TRAIN
  |
  v
SAVE CHECKPOINT
  |
  v
STOP
  |
  v
LOAD CHECKPOINT
  |
  v
RESUME TRAINING
```

A DQN checkpoint should preserve the training state necessary to continue an experiment, including:

- Online network
- Target network
- Optimizer state
- Current episode
- Global step
- Epsilon
- Configuration
- Random-state information where practical

---

# Generated Artifacts

Training and evaluation can generate artifacts such as:

```text
checkpoints/
models/
logs/
evaluations/
reports/
```

Examples include:

```text
models/snake_dqn_best.pt

checkpoints/snake/checkpoint_ep500.pt
checkpoints/snake/checkpoint_ep1000.pt
```

Training logs may contain:

```text
episode
reward
episode_length
loss
epsilon
average Q
score
seed
```

These artifacts are generated during experiments and are not treated as source code.

---

# Git and Generated Files

The repository's `.gitignore` excludes files that should not normally be committed, including:

- Virtual environments
- Python cache files
- Pytest cache
- Generated checkpoints
- Trained model files
- Training logs
- Evaluation outputs
- Generated plots
- Generated videos
- Other experiment artifacts

Directory placeholders can be retained using `.gitkeep` files so that the required directory structure exists in a fresh clone.

This keeps the Git repository focused on source code, configuration, tests, documentation, and reproducible project definitions rather than large generated artifacts.

---

# Reproducibility

Reproducibility is a core requirement of the project.

Experiments should record:

```text
Game
Agent
State representation
Reward configuration
Learning rate
Gamma
Epsilon start
Epsilon end
Epsilon decay
Batch size
Replay configuration
Target update frequency
Training gap
Episode count
Random seed
Environment version
Code version
```

Example Snake configuration:

```yaml
game: snake
agent: refined_dqn

seed: 42

state:
  shape: [4, 64, 64]

optimizer:
  learning_rate: 0.0001

gamma: 0.99

epsilon:
  start: 1.0
  end: 0.02
  decay_frames: 200000

replay:
  high_error_capacity: 35000
  low_error_capacity: 15000
  td_threshold: 0.5
  high_fraction: 0.70
  low_fraction: 0.30

batch_size: 32

target_update_steps: 500

training_gap: 4
```

A fixed seed should be used when an experiment needs to be reproduced.

---

# Development Workflow

The project follows an environment-first development process.

```text
1. Build environment
       |
       v
2. Verify reset
       |
       v
3. Verify actions
       |
       v
4. Verify rewards
       |
       v
5. Verify termination
       |
       v
6. Verify observation dimensions
       |
       v
7. Implement preprocessing
       |
       v
8. Implement agent
       |
       v
9. Run short training
       |
       v
10. Inspect logs
       |
       v
11. Save checkpoint
       |
       v
12. Resume from checkpoint
       |
       v
13. Run longer training
       |
       v
14. Evaluate frozen model
```

The environment and agent are intentionally kept separate.

---

# Validation Principles

The project follows several validation rules.

## Environment First

The environment must reliably support:

```text
reset
  |
  v
observe
  |
  v
action
  |
  v
step
  |
  v
reward + next state + termination
```

Training should not begin before the environment behaves correctly.

## Training and Evaluation Are Separate

Training:

```text
Exploration ON
Learning ON
Parameters changing
```

Evaluation:

```text
Exploration OFF / controlled
Learning OFF
Parameters frozen
```

A training episode should not be presented as a clean evaluation result.

## Never Rely on One Episode

Final evaluation should report statistical measurements such as:

- Mean score
- Standard deviation
- Mean return
- Episode length
- Success rate
- Score distribution
- Learning curves
- Representative gameplay
- Failure cases

A single highest score is not sufficient evidence of successful learning.

---

# Future Roadmap

The complete project roadmap is:

```text
PART 0
Project Setup
    |
    v
PART 1
Common RL Infrastructure
    |
    v
PART 2
Build + Verify Snake
    |
    v
PART 3
Train Snake DQN
    |
    v
PART 4
Build + Verify Flappy Bird
    |
    v
PART 5
Train Flappy Q-Learning + SARSA
    |
    v
PART 6
Build + Verify Pac-Man
    |
    v
PART 7
Train Pac-Man DQN
    |
    v
PART 8
Unified Evaluation
    |
    v
PART 9
Ablation + Failure Analysis
    |
    v
PART 10
Final Plots + Report + Demo
```

## Why this order?

### Snake

Snake provides a relatively simple environment while introducing:

- Visual state
- Temporal frame history
- Collision logic
- Reward design
- Replay
- DQN
- Target networks
- Checkpointing
- Evaluation

### Flappy Bird

Flappy Bird provides a small action space and a compact discrete state representation, making it suitable for comparing:

```text
Q-Learning
      vs
SARSA
```

### Pac-Man

Pac-Man introduces a more complicated visual environment and a larger strategic state space.

It is therefore scheduled after the team has completed the complete Snake DQN pipeline.

---

# Planned Flappy Bird Stage

The planned Flappy Bird implementation will use:

```text
Q-Learning
+
SARSA
```

with a project-defined discrete state representation of approximately 75 states.

The purpose is to compare:

```text
Q-Learning
    |
    | off-policy
    v
Optimal-policy target

        VS

SARSA
    |
    | on-policy
    v
Behavior-policy target
```

The comparison will use the same environment and state representation wherever possible.

---

# Planned Pac-Man Stage

The planned Pac-Man implementation will use a DQN operating on visual observations.

Target representation:

```text
Screen
  |
  v
Preprocessing
  |
  v
84 × 84 frame
  |
  v
4-frame history
  |
  v
(4, 84, 84)
  |
  v
CNN
  |
  v
Dueling DQN
  |
  v
Action
```

The intended environment basis is Atari Ms. Pac-Man.

If the project uses a different Pac-Man implementation, that difference must be explicitly documented rather than being presented as the exact paper environment.

---

# Evaluation Methodology

The final project is intended to use controlled evaluation rather than isolated gameplay demonstrations.

For each completed agent:

```text
TRAINED MODEL
     |
     v
FROZEN AGENT
     |
     v
MULTIPLE EVALUATION EPISODES
     |
     +--> Score
     +--> Return
     +--> Episode length
     +--> Success rate
     +--> Distribution
     +--> Failure cases
     |
     v
COMPARISON
```

A random policy should also be used as a baseline where appropriate.

The final analysis should compare:

- Agent performance
- Learning curves
- Stability
- Variance
- Survival
- Game-native score
- Failure behavior

---

# Planned Ablation Studies

After the primary agents are working, controlled ablations can investigate the contribution of individual design choices.

Potential factors include:

- Reward shaping
- Frame stacking
- Replay strategy
- Exploration schedule
- Target network frequency
- State representation
- Network architecture

For example:

```text
Baseline DQN
     |
     +----> + Dueling Head
     |
     +----> + Dual Replay
     |
     +----> + Frame History
     |
     +----> + Reward Shaping
```

Each modification should be documented separately.

---

# Research Basis

The project is based on the supplied project milestones and selected reinforcement learning literature.

## General Reinforcement Learning

- Watkins & Dayan — Q-Learning
- Rummery & Niranjan — SARSA
- Sutton & Barto — Reinforcement Learning

## Deep Reinforcement Learning

- Mnih et al. — *Playing Atari with Deep Reinforcement Learning*
- Wang et al. — *Dueling Network Architectures for Deep Reinforcement Learning*

## Snake

- Zhepei Wei et al. — *Autonomous Agents in Snake Game via Deep Reinforcement Learning*

## Flappy Bird

- Tai Vu & Leon Tran — *FlapAI Bird: Training an Agent to Play Flappy Bird Using Reinforcement Learning Techniques*

## Pac-Man

- K. Vijaychandra Reddy — *Playing Pac-Man using Reinforcement Learning*

The official project Milestone 3 configuration is treated as the controlling specification whenever it differs from a research paper's implementation details.

---

# External Resources

## Gymnasium

Official documentation:

https://gymnasium.farama.org/

Official repository:

https://github.com/Farama-Foundation/Gymnasium

Used for:

- Environment API
- `reset()` / `step()`
- Spaces
- Wrappers
- Environment debugging
- Compatibility

## Atari / ALE

Gymnasium Atari documentation:

https://gymnasium.farama.org/environments/atari/

Arcade Learning Environment:

https://github.com/Farama-Foundation/Arcade-Learning-Environment

ALE research:

https://arxiv.org/abs/1207.4708

## Flappy Bird Gymnasium

https://github.com/markub3327/flappy-bird-gymnasium

Used for the planned Flappy Bird environment integration.

## DQN

https://arxiv.org/abs/1312.5602

## Dueling DQN

https://arxiv.org/abs/1511.06581

---

# Important Scope Notes

This repository should not make claims that exceed the experimental evidence.

The project does **not** claim:

- Universal game-playing intelligence
- Human-level performance
- General intelligence across games
- That DQN is on-policy
- That one successful episode proves learning

Such claims would require additional controlled experiments and evidence.

---

# Current Usage Summary

For the current completed Snake implementation:

### Test

```powershell
python -m pytest tests/
```

### Verbose test

```powershell
pytest tests/ -v
```

### Train

```powershell
python train_snake.py
```

### Train with configuration

```powershell
python train_snake.py --config configs/snake_dqn.yaml --episodes 1000 --seed 42
```

### Resume training

```powershell
python train_snake.py --resume checkpoints/snake/checkpoint_ep500.pt
```

### Evaluate best model

```powershell
python evaluate_snake.py --model models/snake_dqn_best.pt --episodes 100
```

### Evaluate checkpoint

```powershell
python evaluate_snake.py --model checkpoints/snake/checkpoint_ep1000.pt --episodes 100
```

---

# Project Deliverables

## Current Snake Deliverables

```text
[✓] Snake environment
[✓] Environment validation
[✓] Snake DQN
[✓] Dueling network
[✓] Dual replay
[✓] Target network
[✓] Epsilon-greedy exploration
[✓] Training pipeline
[✓] Checkpointing
[✓] Checkpoint resuming
[✓] Training logging
[✓] Evaluation pipeline
[✓] Automated tests
```

## Final Project Deliverables

The completed project is intended to contain:

```text
[ ] Snake refined DQN
[ ] Flappy Bird Q-Learning
[ ] Flappy Bird SARSA
[ ] Pac-Man DQN

[ ] Random baselines
[ ] 100-episode evaluations
[ ] Learning curves
[ ] Evaluation distributions
[ ] Ablation experiments
[ ] Failure analysis
[ ] Gameplay demonstrations
[ ] Final report
[ ] Final comparison
[ ] Reproducibility documentation
```

---

# Project Philosophy

The project follows a simple dependency chain:

```text
ENVIRONMENT
     |
     v
VALIDATION
     |
     v
STATE REPRESENTATION
     |
     v
AGENT
     |
     v
TRAINING
     |
     v
CHECKPOINT
     |
     v
EVALUATION
     |
     v
ANALYSIS
     |
     v
CONCLUSION
```

Each stage should be verified before conclusions are drawn from the next stage.

The goal is not merely to produce a game-playing demonstration. The goal is to build a reproducible reinforcement learning pipeline in which the relationship between **environment design, state representation, reward design, learning algorithm, training strategy, and measured performance** can be examined systematically.

---

# License

This project is developed for academic and educational purposes.

Add the appropriate license here if the repository is formally released under a specific open-source license.