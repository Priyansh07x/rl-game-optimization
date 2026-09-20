# Cornerstone Project --- Correct Implementation Plan

## Purpose

This document restructures the implementation workflow of the project
into a dependency-correct sequence.

The main principle is:

``` text
SETUP
  |
  v
COMMON INFRASTRUCTURE
  |
  v
SNAKE ENVIRONMENT
  |
  v
SNAKE DQN
  |
  v
SNAKE EVALUATION
  |
  v
FLAPPY ENVIRONMENT
  |
  v
FLAPPY Q-LEARNING + SARSA
  |
  v
FLAPPY EVALUATION
  |
  v
PAC-MAN ENVIRONMENT
  |
  v
PAC-MAN DQN
  |
  v
PAC-MAN EVALUATION
  |
  v
UNIFIED BASELINES + COMPARISON
  |
  v
ABLATION + FAILURE ANALYSIS
  |
  v
FINAL PLOTS + VIDEOS + REPORT
```

This ordering fixes the inconsistent section numbering in the original
guide. The practical roadmap is treated as authoritative for
implementation order.

------------------------------------------------------------------------

# Part 0 --- Project Setup

## Goal

Establish the repository, Python environment, dependencies, folder
structure, and basic reproducibility infrastructure.

## Tasks

1.  Confirm the project repository.
2.  Create/activate the Python virtual environment.
3.  Install core dependencies:
    -   Python
    -   NumPy
    -   PyTorch
    -   Gymnasium
    -   Matplotlib
    -   Pandas
4.  Install additional game-specific dependencies only when the
    corresponding game is started.
5.  Establish Git tracking.
6.  Create the project folder structure.
7.  Create initial configuration, logging, metadata, and testing
    infrastructure.
8.  Run a basic Gymnasium smoke test.

## Validation gate

Do not proceed until:

``` text
Python environment works
      +
core dependencies import
      +
Gymnasium environment can reset
      +
Gymnasium environment can step
      +
Git repository is usable
```

------------------------------------------------------------------------

# Part 1 --- Common RL Infrastructure

## Goal

Build reusable infrastructure before implementing game-specific agents.

## Components

``` text
src/
├── common/
│   ├── seed.py
│   ├── logger.py
│   ├── metrics.py
│   ├── checkpoint.py
│   ├── evaluation.py
│   └── utils.py
│
├── environments/
│   └── base_env.py
│
├── preprocessing/
│   └── frame_stack.py
│
└── agents/
    ├── tabular/
    └── dqn/
```

## Tasks

### 1. Common environment contract

Use the conceptual interface:

``` text
reset()
   |
   v
observation

step(action)
   |
   +--> next observation
   +--> reward
   +--> terminated/truncated
   +--> info
```

Gymnasium's modern API must distinguish `terminated` and `truncated`.

### 2. Seed management

Provide deterministic seed handling where practical.

### 3. Logging

Create a common mechanism for recording:

``` text
episode
reward
episode_length
loss
epsilon
score
seed
```

DQN experiments should additionally record average Q where applicable.

### 4. Checkpoint utilities

Support saving/loading the information required by the relevant agent.

### 5. Metrics utilities

Provide reusable calculations for:

-   mean
-   standard deviation
-   moving averages
-   success rate
-   episode statistics

### 6. Testing foundation

Create the basic test structure before game-specific development.

## Validation gate

``` text
common utilities tested
       +
seed handling verified
       +
logging verified
       +
checkpoint save/load verified
       +
tests execute successfully
```

------------------------------------------------------------------------

# Part 2 --- Snake Environment

## Goal

Build and completely verify the Snake--Apple environment before relying
on it for DQN training.

## Environment specification

The supplied project material describes:

``` text
Map:              240 × 240 pixels
Logical grid:     12 × 12
Initial length:   3
Initial direction:right
Apple:            one at a time
Controls:         UP / DOWN / LEFT / RIGHT
Termination:      wall collision / self collision
```

## Action mapping

``` text
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

Keep this mapping in one documented configuration/source location.

## Reward boundary

Keep the research-paper reward description separate from the official
Milestone 3 training configuration.

For the official refined Snake DQN training configuration:

``` text
apple      = +1.0
survival   = +0.1
collision  = -1.0
```

## Observation pipeline

The intended conceptual pipeline is:

``` text
240 × 240 game
      |
      v
12 × 12 logical grid
      |
      v
rendered RGB frame
      |
      v
64 × 64 preprocessing
      |
      v
processed frame
      |
      v
4-frame history
      |
      v
DQN state
```

The implementation must explicitly document the tensor layout because
the source material uses both `(4, 64, 64)` and an intermediate
`64 × 64 × 12` description.

## Validation checklist

``` text
[ ] reset works
[ ] snake length = 3
[ ] initial direction = right
[ ] valid snake/apple placement
[ ] apple exists
[ ] movement works
[ ] all four actions map correctly
[ ] wall collision terminates
[ ] self collision terminates
[ ] apple reward works
[ ] observation shape is correct
[ ] frame stack is correct
[ ] terminal reset works
[ ] random-agent integration works
```

## Evidence

Produce:

``` text
reports/environment_validation.md
logs/snake_environment_smoke_test.csv
observations/snake/
reward_traces/snake/
episode_traces/snake/
```

## Validation gate

Do not implement or scale Snake DQN until every environment validation
item passes.

------------------------------------------------------------------------

# Part 3 --- Snake Refined DQN

## Goal

Implement, train, checkpoint, and formally evaluate the Snake DQN.

## DQN configuration

Official project configuration:

``` text
Input:
    (4, 64, 64)

Convolution:
    16 filters
    32 filters
    64 filters

Fully connected:
    256 units

Head:
    Dueling DQN
```

## Replay

``` text
High-TD-error capacity: 35,000
Low-TD-error capacity:  15,000
Total:                   50,000

TD threshold:            0.5

Sampling:
    70% high-error
    30% low-error
```

## Training

``` text
Batch size:          32
Learning rate:       0.0001
Gamma:               0.99
Target update:       every 500 steps
Training gap:        4 frames
Epsilon:             1.0 -> 0.02
Epsilon decay:       200,000 frames
```

## Implementation order

``` text
Dueling network
      |
      v
Replay buffers
      |
      v
Target network
      |
      v
Epsilon-greedy
      |
      v
Training loop
      |
      v
Logging
      |
      v
Checkpointing
```

## Mandatory tests

Before long training:

``` text
[ ] network forward pass
[ ] replay insertion
[ ] replay sampling
[ ] TD-error classification
[ ] target calculation
[ ] terminal-state handling
[ ] one-batch training update
[ ] finite loss
[ ] gradients exist
[ ] weights change
[ ] checkpoint save
[ ] checkpoint load
```

## Training progression

``` text
short training run
      |
      v
inspect logs
      |
      v
checkpoint/resume test
      |
      v
longer training
      |
      v
freeze selected checkpoint
```

## Evaluation

Evaluation must use:

``` text
learning OFF
parameters frozen
exploration disabled/controlled
fixed evaluation protocol
```

The official initial evaluation target is:

``` text
100 evaluation episodes
```

Record:

``` text
episode
score
return
episode length
success indicator
seed
```

## Snake completion gate

Snake is considered complete for the implementation stage only when:

``` text
environment verified
      +
DQN verified
      +
training completed
      +
checkpoint loads
      +
evaluation completed
      +
evaluation artifacts saved
```

Project-wide baselines, repeated seeds, ablations, and final comparisons
remain later tasks.

------------------------------------------------------------------------

# Part 4 --- Flappy Bird Environment

## Goal

Build and verify the Flappy Bird environment independently before
implementing Q-Learning or SARSA.

## Environment basis

Use the selected Flappy Bird Gymnasium environment and explicitly
document the exact environment/version used.

Do not assume that the repository's default reward values equal the
project's reward configuration.

## Raw state information

The project state representation uses:

``` text
horizontal distance
vertical distance
bird y-velocity
```

## Discretization

Official project representation:

``` text
x bins        = 5
y bins        = 5
velocity bins = 3

total states = 5 × 5 × 3
             = 75
```

State:

``` text
(x_bin, y_bin, velocity_bin)
```

## Actions

``` text
0 = flap
1 = do not flap
```

## Reward

Official project configuration:

``` text
survive     = +0.5
pass pipe   = +5
collision   = -1000
```

## Validation checklist

``` text
[ ] environment starts
[ ] reset works
[ ] horizontal distance available
[ ] vertical distance available
[ ] velocity available
[ ] discretization works
[ ] exactly 75 state combinations are representable
[ ] action 0 = flap
[ ] action 1 = no flap
[ ] survival reward works
[ ] pipe-pass reward works
[ ] collision reward works
[ ] terminal state works
[ ] random-agent integration works
```

## Evidence

Produce:

``` text
logs/flappy_environment_smoke_test.csv
observations/flappy/
reward_traces/flappy/
episode_traces/flappy/
```

## Validation gate

Do not begin tabular training until the environment passes all
validation tests.

------------------------------------------------------------------------

# Part 5 --- Flappy Bird Q-Learning + SARSA

## Goal

Implement the two tabular algorithms under the same environment, state
representation, actions, rewards, and evaluation protocol.

## Q-table

Use:

``` text
75 states × 2 actions
```

Conceptually:

``` text
Q[state][action]
```

## Q-Learning

Q-Learning is off-policy.

``` text
Q(s,a) <- Q(s,a)
          + alpha[
              r + gamma * max Q(s',a')
              - Q(s,a)
            ]
```

## SARSA

SARSA is on-policy.

``` text
Q(s,a) <- Q(s,a)
          + alpha[
              r + gamma * Q(s',a')
              - Q(s,a)
            ]
```

The next action must be selected using the behavior policy.

## Epsilon schedule

Official project configuration:

``` text
epsilon: 1.0 -> 0.05
decay:   5,000 episodes
```

## Fair comparison requirement

Q-Learning and SARSA must use:

``` text
same environment
same state encoder
same actions
same reward function
same evaluation horizon
same evaluation seeds
same scoring criteria
```

Only the learning algorithm should differ in the main comparison.

## Checkpointing

Tabular checkpoints should preserve:

``` text
Q-table
episode
epsilon
configuration
seed
```

## Logging

Record at minimum:

``` text
episode
reward
episode_length
epsilon
score
q_table_size
seed
```

## Validation gate

Before long runs:

``` text
[ ] Q-table indexing works
[ ] Q-Learning update verified
[ ] SARSA update verified
[ ] terminal handling verified
[ ] epsilon schedule verified
[ ] checkpoint save/load verified
[ ] short training run works
```

------------------------------------------------------------------------

# Part 6 --- Flappy Bird Evaluation

## Goal

Evaluate Q-Learning and SARSA using frozen tabular policies.

## Evaluation

For each agent:

``` text
load checkpoint
      |
      v
freeze Q-table
      |
      v
disable exploration
      |
      v
run fixed evaluation episodes
      |
      v
save evaluation CSV
```

Use the common evaluation protocol, including the official 100-episode
initial evaluation where applicable.

## Outputs

``` text
models/flappy_q_table.pkl
models/flappy_sarsa_table.pkl

evaluations/flappy/
├── q_learning.csv
└── sarsa.csv
```

Then produce within-game comparison:

``` text
Random
   |
   +--> Q-Learning
   |
   +--> SARSA
```

Do not rank algorithms by a single lucky episode.

------------------------------------------------------------------------

# Part 7 --- Pac-Man Environment

## Goal

Build and verify the Atari 2600 Ms. Pac-Man environment before DQN
development.

## Environment boundary

The supplied research basis is specifically:

``` text
Atari 2600 Ms. Pac-Man
```

If a different Pac-Man implementation is used, document it explicitly as
a project modification.

## Observation pipeline

``` text
Atari observation
      |
      v
configured crop/preprocessing
      |
      v
84 × 84 frame
      |
      v
grayscale / binary-style representation
      |
      v
4-frame stack
      |
      v
(4, 84, 84)
```

The exact preprocessing must be documented rather than silently
substituted.

## Actions

Use the legal controller actions exposed by the selected environment.

Document the action mapping in:

``` text
configs/pacman/env.yaml
```

Do not assume action indices are identical across Atari wrappers.

## Validation checklist

``` text
[ ] Atari environment installs
[ ] Ms. Pac-Man environment starts
[ ] reset works
[ ] observation is valid
[ ] preprocessing works
[ ] 84 × 84 representation is correct
[ ] frame stack is correct
[ ] action mapping is documented
[ ] reward/termination information is valid
[ ] random-agent integration works
```

## Evidence

Produce:

``` text
logs/pacman_environment_smoke_test.csv
observations/pacman/
reward_traces/pacman/
episode_traces/pacman/
```

## Validation gate

Do not implement Pac-Man DQN until the environment and preprocessing
pass validation.

------------------------------------------------------------------------

# Part 8 --- Pac-Man DQN

## Goal

Implement and train the official project Pac-Man DQN.

## Network

Official project configuration:

``` text
Input:
    (4, 84, 84)

CNN:
    32 -> 64 -> 64 channels

Kernel:
    3 × 3

Stride:
    1

Fully connected:
    512

Head:
    Dueling

Actions:
    4
```

## Replay and optimization

``` text
Replay capacity:    100,000
Batch size:         32
Learning rate:      0.00025
Gamma:              0.99
Target update:      every 1,000 steps
```

## Epsilon

``` text
epsilon: 1.0 -> 0.01
decay:   500,000 frames
```

## Reward

Official project configuration:

``` text
pellet = +0.1
power  = +0.5
step   = -0.01
ghost  = -50
```

## Training order

``` text
preprocessing verified
      |
      v
CNN verified
      |
      v
dueling head verified
      |
      v
replay verified
      |
      v
target network verified
      |
      v
one-batch update
      |
      v
short training
      |
      v
checkpoint/resume
      |
      v
longer training
```

## Completion gate

Do not move to unified experiments until:

``` text
Pac-Man environment verified
      +
DQN update verified
      +
training completed
      +
checkpoint loads
      +
initial evaluation completed
```

------------------------------------------------------------------------

# Part 9 --- Unified Evaluation + Random Baselines

## Goal

Establish the common evaluation framework after all three game pipelines
are independently functional.

## Random baselines

For each game:

``` text
random policy
     |
     v
multiple episodes
     |
     v
baseline distribution
```

Store:

``` text
evaluations/<game>/random_baseline.csv
```

## Main experiment matrix

``` text
Game          Agent
--------------------------------
Snake         Refined DQN
Flappy Bird   Q-Learning
Flappy Bird   SARSA
Pac-Man       DQN
```

## Evaluation metrics

For every applicable game:

``` text
mean score
standard deviation
mean return
episode length / survival
success rate
score distribution
```

For DQN:

``` text
loss
average Q
epsilon during training
```

For Flappy:

``` text
Q-table size
```

## Evaluation rule

Never compare raw game scores across different games as though they
share a common scale.

Compare agents within the same game.

------------------------------------------------------------------------

# Part 10 --- Visualization

## Goal

Turn raw training/evaluation data into reproducible figures.

## Minimum plots

``` text
1. reward vs episode
2. moving-average reward
3. episode length vs episode
4. score vs episode
5. loss vs training step/episode
6. epsilon vs episode/frame
7. evaluation score distribution
8. algorithm comparison
```

Additional:

``` text
DQN:
    average Q

Flappy:
    Q-table size
```

## Rule

Never replace raw data with a moving average.

Store:

``` text
raw data
+
derived visualization data
```

------------------------------------------------------------------------

# Part 11 --- Gameplay Recording

## Goal

Create visual evidence of learned and failed behavior.

For each game, where applicable, record:

``` text
random baseline
early training
mid training
final trained agent
failure case
```

Suggested structure:

``` text
videos/
├── snake/
├── flappy/
└── pacman/
```

Gameplay videos are evidence, not substitutes for quantitative
evaluation.

------------------------------------------------------------------------

# Part 12 --- Controlled Ablation Studies

## Goal

Test individual design choices while changing only the selected factor.

## Candidate ablations

### Reward ablation

``` text
baseline reward
      vs
modified reward
```

### State ablation

``` text
4-frame history
      vs
single frame
```

### Exploration ablation

Compare documented epsilon schedules.

### Replay ablation

For Snake:

``` text
dual replay
      vs
ordinary replay
```

### Algorithm ablation

For Flappy:

``` text
Q-Learning
      vs
SARSA
```

## Rule

An ablation is valid only if:

``` text
one factor changes
all other relevant conditions remain controlled
```

Record the exact configuration for every ablation.

------------------------------------------------------------------------

# Part 13 --- Failure Analysis

## Goal

Explain failures rather than merely reporting that an agent performed
poorly.

Use this diagnostic order:

``` text
FAILURE
   |
   +--> environment bug?
   |
   +--> state representation?
   |
   +--> reward problem?
   |
   +--> exploration problem?
   |
   +--> unstable DQN?
   |
   +--> insufficient training?
   |
   +--> game-specific difficulty?
```

## DQN diagnostics

Inspect:

``` text
replay buffer
target update
learning rate
reward scaling
gradient handling
state normalization
terminal handling
loss
Q-values
```

## Reward-hacking diagnostics

Inspect all of:

``` text
native score
reward
trajectory
behavior
```

Do not infer successful learning from reward alone.

------------------------------------------------------------------------

# Part 14 --- Reproducibility + Experiment Registry

## Goal

Make every reported experiment traceable.

## Record

``` text
Python version
package versions
environment ID
environment version/commit where practical
state definition
preprocessing
action mapping
reward constants
alpha
gamma
epsilon schedule
episode count
frame count
seed
network architecture
learning rate
replay configuration
target update interval
checkpoint interval
evaluation protocol
code version
```

## Seeds

Use multiple seeds where feasible.

Example:

``` text
42
123
456
789
2026
```

Store them in:

``` text
metadata/seeds.json
```

## Experiment IDs

Examples:

``` text
SNAKE_DQN_SEED42_V1
FLAPPY_Q_SEED42_V1
FLAPPY_SARSA_SEED42_V1
PACMAN_DQN_SEED42_V1
```

Maintain:

``` text
metadata/experiment_registry.json
```

------------------------------------------------------------------------

# Part 15 --- Final Comparison

## Goal

Produce evidence-based conclusions from the completed experiments.

## Within-game comparisons

``` text
Snake:
    Random vs DQN

Flappy:
    Random vs Q-Learning vs SARSA

Pac-Man:
    Random vs DQN
```

## Cross-game analysis

Do not compare raw scores across games.

Instead compare qualitative/experimental characteristics:

``` text
state complexity
training stability
sample efficiency
reward sparsity
exploration sensitivity
failure modes
```

------------------------------------------------------------------------

# Part 16 --- Final Report + Demonstration

## Goal

Package the entire project into a reproducible final deliverable.

## Report structure

``` text
1. Introduction
2. Problem Statement
3. Objectives
4. Reinforcement Learning Background
5. MDP Formulation
6. Common System Architecture
7. Environment Design
   7.1 Snake
   7.2 Flappy Bird
   7.3 Pac-Man
8. State Representation
9. Reward Design
10. Algorithms
    10.1 Q-Learning
    10.2 SARSA
    10.3 DQN
11. Implementation
12. Training Configuration
13. Experimental Setup
14. Baselines
15. Evaluation Metrics
16. Results
17. Ablation Studies
18. Failure Analysis
19. Reproducibility
20. Limitations
21. Conclusion
22. References
```

## Demonstration

### Snake

``` text
launch
  |
  v
load trained DQN
  |
  v
automatic gameplay
  |
  v
score + behavior
```

### Flappy

``` text
Q-Learning
    |
    v
gameplay

SARSA
    |
    v
gameplay

comparison
```

### Pac-Man

``` text
load trained DQN
    |
    v
processed observation
    |
    v
action selection
    |
    v
gameplay
```

### Evaluation dashboard

Show:

``` text
mean score
std score
mean return
survival
success rate
training curves
evaluation distributions
algorithm comparisons
```

------------------------------------------------------------------------

# Final Definition of Done

The entire project is complete only when:

``` text
All environments work
        +
State representations are verified
        +
Rewards are verified
        +
All required agents train
        +
Checkpoints work
        +
Logs exist
        +
Random baselines exist
        +
Repeated evaluation exists
        +
Results are reproducible
        +
Failure cases are analyzed
        +
Ablations are documented
        +
Comparisons are fair
        +
Plots exist
        +
Gameplay evidence exists
        +
Final report exists
```

------------------------------------------------------------------------

# Current Execution Position

Because Snake Part 3 has already been completed, the working sequence
from this point is:

``` text
PART 0  SETUP                         [COMPLETED]
   |
PART 1  COMMON INFRASTRUCTURE        [COMPLETED]
   |
PART 2  SNAKE ENVIRONMENT             [COMPLETED]
   |
PART 3  SNAKE DQN + EVALUATION        [COMPLETED]
   |
   v
PART 4  FLAPPY BIRD ENVIRONMENT       <-- CURRENT
   |
PART 5  FLAPPY Q-LEARNING + SARSA
   |
PART 6  FLAPPY EVALUATION
   |
PART 7  PAC-MAN ENVIRONMENT
   |
PART 8  PAC-MAN DQN
   |
PART 9  UNIFIED EVALUATION
   |
PART 10 VISUALIZATION
   |
PART 11 GAMEPLAY RECORDING
   |
PART 12 ABLATIONS
   |
PART 13 FAILURE ANALYSIS
   |
PART 14 REPRODUCIBILITY
   |
PART 15 FINAL COMPARISON
   |
PART 16 FINAL REPORT + DEMO
```

## Important numbering rule

The original guide contains later headings whose Part numbers conflict
with its practical roadmap. For implementation, use **this document's
numbering**.

Therefore:

> **Part 4 means Flappy Bird Environment Build + Verification.**

Do not begin Flappy Q-Learning/SARSA until Part 4's validation gate has
passed.

Do not begin Pac-Man until the complete Flappy pipeline has been built,
trained, evaluated, and validated.
