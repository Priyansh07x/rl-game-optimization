# Project Resources

## PS 19: Implement Reinforcement Learning for Game Optimization

This file contains the recommended links for the project covering the RL
framework, game environments, DQN research, and project-specific
background material.

> **Important:** These are primarily **framework, environment,
> implementation, and research resources**. This project does not
> require a conventional static dataset such as a Kaggle dataset. The RL
> agents generate their experience online by interacting with the game
> environments.

------------------------------------------------------------------------

## 1. Core RL Framework

### 1.1 Gymnasium Documentation --- REQUIRED

**Link:** https://gymnasium.farama.org/

**Use for:** - Understanding the modern Gymnasium API. - `reset()` and
`step()` environment interaction. - Observation and action spaces. -
Wrappers and environment preprocessing. - Building and validating custom
environments. - Recording and evaluation utilities.

**Project relevance:** **Very High**

**Used in:** All three games / environment layer.

------------------------------------------------------------------------

### 1.2 Gymnasium GitHub Repository --- REQUIRED

**Link:** https://github.com/Farama-Foundation/Gymnasium

**Use for:** - Official source code. - Installation and dependency
information. - Environment implementations. - Examples and issue
tracking. - Checking compatibility when debugging.

**Project relevance:** **High**

**Used in:** Project infrastructure and environment debugging.

------------------------------------------------------------------------

## 2. Pac-Man / Atari Resources

### 2.1 Gymnasium Atari Documentation --- REQUIRED

**Link:** https://gymnasium.farama.org/environments/atari/

**Use for:** - Atari environment setup. - Atari observation and action
spaces. - Frame handling. - Available Atari environments. -
Understanding the environment interface.

**Project relevance:** **Very High**

**Used in:** Pac-Man implementation.

> **Important implementation note:** The Milestone 3 Pac-Man input is
> `(4, 84, 84)`. This is the project's processed state representation,
> not the raw Gymnasium Atari observation. The implementation needs
> preprocessing such as grayscale conversion, resizing to `84x84`, and
> stacking four frames.

------------------------------------------------------------------------

### 2.2 Arcade Learning Environment (ALE) Pac-Man Documentation --- REQUIRED

**Link:**
https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/docs/environments/pacman.md

**Use for:** - Pac-Man environment details. - Available Pac-Man
variants. - Observation/action information. - Atari-specific behavior
and configuration.

**Project relevance:** **Very High**

**Used in:** Pac-Man environment verification.

> The project source material specifically refers to Atari 2600
> Ms. Pac-Man. If another Pac-Man implementation is used, document that
> as a project modification.

------------------------------------------------------------------------

### 2.3 Arcade Learning Environment Paper --- RECOMMENDED

**Paper:** *The Arcade Learning Environment: An Evaluation Platform for
General Agents*

**Link:** https://arxiv.org/abs/1207.4708

**Use for:** - Background on Atari-based RL evaluation. - Understanding
the ALE platform. - Research context for game-playing agents.

**Project relevance:** **High**

**Used in:** Literature review and Pac-Man/Atari background.

------------------------------------------------------------------------

## 3. Flappy Bird Resources

### 3.1 Flappy Bird Gymnasium Environment --- REQUIRED

**Repository:** `markub3327/flappy-bird-gymnasium`

**Link:** https://github.com/markub3327/flappy-bird-gymnasium

**Use for:** - Running a Gymnasium-compatible Flappy Bird environment. -
Understanding the available actions. - Understanding environment
observations. - Game mechanics and environment integration.

**Project relevance:** **Very High**

**Used in:** Flappy Bird environment implementation.

### Reward compatibility warning

The repository documents its own environment rewards, but the official
Milestone 3 project configuration uses a different reward scheme:

  Event         Project reward
  ----------- ----------------
  Survive                 +0.5
  Pass pipe                 +5
  Collision              -1000

Therefore:

-   Use the repository for the **environment/game mechanics**.
-   Do **not** automatically copy its reward values into the project.
-   Implement the Milestone 3 reward function/wrapper explicitly.
-   Document any environment or reward modification.

------------------------------------------------------------------------

## 4. DQN Research

### 4.1 Mnih et al. --- Playing Atari with Deep Reinforcement Learning --- REQUIRED

**Paper:** *Playing Atari with Deep Reinforcement Learning*

**Link:** https://arxiv.org/abs/1312.5602

**Use for:** - DQN background. - Learning from raw visual game
observations. - CNN-based function approximation. - Experience replay. -
Atari game-playing methodology.

**Project relevance:** **Very High**

**Used in:** Pac-Man DQN and Snake DQN background.

------------------------------------------------------------------------

### 4.2 Dueling Network Architectures for Deep Reinforcement Learning --- REQUIRED

**Paper:** *Dueling Network Architectures for Deep Reinforcement
Learning*

**Link:** https://arxiv.org/abs/1511.06581

**Use for:** - Understanding dueling DQN architecture. - Value and
advantage streams. - Why a dueling head can be useful when actions have
similar effects.

**Project relevance:** **Very High**

**Used in:** Pac-Man DQN and Snake refined DQN, because Milestone 3
specifies a dueling head.

------------------------------------------------------------------------

## 5. Project-Specific Research Resources

These resources should be selected from the project's Milestone 1--3
literature/reference material and used for the literature review,
implementation justification, and comparison methodology.

### 5.1 Pac-Man RL Research

**Purpose:** - Pac-Man state representation. - Reward design. -
Exploration strategies. - Atari/game-playing RL approaches.

**Project relevance:** **High**

**Use:** Literature review and design justification.

> Add the exact paper URL from the project's approved
> literature/reference list here if one has been selected. The current
> links file did not provide a specific Pac-Man research-paper URL.

------------------------------------------------------------------------

### 5.2 Snake RL / DQN Research

**Purpose:** - Snake state representation. - Reward shaping. - DQN-based
Snake agents. - Game-specific learning challenges.

**Project relevance:** **High**

**Use:** Literature review and Snake design justification.

> Add the exact paper URL from the project's approved
> literature/reference list here if one has been selected. The current
> links file did not provide a specific Snake research-paper URL.

------------------------------------------------------------------------

### 5.3 Flappy Bird RL Research

**Purpose:** - Flappy Bird state representation. - Q-Learning/SARSA
approaches. - Reward design. - Discrete state construction.

**Project relevance:** **High**

**Use:** Literature review and Flappy Bird design justification.

> Add the exact paper URL from the project's approved
> literature/reference list here if one has been selected. The current
> links file did not provide a specific Flappy Bird paper URL.

------------------------------------------------------------------------

## 6. Optional Development Resource

### 6.1 CartPole --- OPTIONAL Smoke Test

**Link:**
https://gymnasium.farama.org/environments/classic_control/cart_pole/

**Use for:** - Testing Gymnasium installation. - Testing the basic RL
training loop. - Debugging epsilon-greedy action selection. - Verifying
logging and checkpointing before running the game environments.

**Project relevance:** **Medium / Development Only**

CartPole should **not** be presented as part of the final three-game
experiment. It is only a convenient smoke test for the implementation
infrastructure.

------------------------------------------------------------------------

## 7. Resources Not Required for the Current Project

The following links were present in the original useful-links file but
are not part of the core implementation because they focus on **offline
RL** or **continuous-action Atari**, while this project is primarily an
online RL project with discrete actions.

### 7.1 Minari --- NOT REQUIRED

**Link:** https://github.com/Farama-Foundation/Minari

**Purpose:** Offline RL dataset and dataset-management ecosystem.

**Why excluded from core project:** - The project does not require a
pre-collected offline RL dataset. - Agents generate experience through
online interaction with the environments.

**Status:** Optional future/offline-RL resource.

------------------------------------------------------------------------

### 7.2 D4RL --- NOT REQUIRED

**Paper:** *D4RL: Datasets for Deep Data-Driven Reinforcement Learning*

**Link:** https://arxiv.org/abs/2004.07219

**Purpose:** Offline RL benchmark datasets.

**Why excluded from core project:** - The current project does not use
offline RL. - D4RL is not needed for the Milestone 3 training pipeline.

**Status:** Optional future/offline-RL resource.

------------------------------------------------------------------------

### 7.3 CALE --- NOT REQUIRED

**Paper:** *Continuous Arcade Learning Environment*

**Link:** https://arxiv.org/abs/2410.23810

**Purpose:** Continuous-action extensions of Atari environments.

**Why excluded from core project:** - The current project uses discrete
game actions. - The Milestone 3 agents are DQN, Q-Learning, and SARSA. -
Continuous-control algorithms are outside the current scope.

**Status:** Optional future research resource.

------------------------------------------------------------------------

## 8. Recommended Link Priority

  Resource                     Priority       Main purpose
  ---------------------------- -------------- -------------------------------
  Gymnasium Documentation      REQUIRED       RL environment API
  Gymnasium GitHub             REQUIRED       Implementation/debugging
  Gymnasium Atari Docs         REQUIRED       Pac-Man/Atari
  ALE Pac-Man Docs             REQUIRED       Pac-Man environment details
  Flappy Bird Gymnasium        REQUIRED       Flappy Bird environment
  Mnih et al. DQN              REQUIRED       DQN foundation
  Dueling DQN paper            REQUIRED       Dueling architecture
  ALE paper                    RECOMMENDED    Atari/ALE research background
  Pac-Man research paper       RECOMMENDED    Game-specific literature
  Snake research paper         RECOMMENDED    Game-specific literature
  Flappy Bird research paper   RECOMMENDED    Game-specific literature
  CartPole                     OPTIONAL       Development smoke test
  Minari                       NOT REQUIRED   Offline RL
  D4RL                         NOT REQUIRED   Offline RL datasets
  CALE                         NOT REQUIRED   Continuous Atari

------------------------------------------------------------------------

## 9. How These Resources Map to the Project

### Pac-Man

``` text
Gymnasium
    ↓
Atari Documentation
    ↓
ALE Pac-Man Documentation
    ↓
ALE Research Paper
    ↓
DQN Paper
    ↓
Dueling DQN Paper
    ↓
Pac-Man DQN Implementation
```

### Snake

``` text
Gymnasium
    ↓
DQN Paper
    ↓
Dueling DQN Paper
    ↓
Snake-specific Research
    ↓
Snake Refined DQN Implementation
```

### Flappy Bird

``` text
Gymnasium
    ↓
Flappy Bird Gymnasium Environment
    ↓
Flappy Bird Research
    ↓
Q-Learning + SARSA
    ↓
75-State Discrete Representation
```

------------------------------------------------------------------------

## 10. Important Project Terminology

Do **not** label this file as simply:

> Dataset Links

A better title is:

> **Project Resources**

or:

> **Project Environment, Implementation and Research Links**

The project uses online reinforcement learning. During training, the
agent generates experience in the form:

``` text
(state, action, reward, next_state, done)
```

For DQN, these transitions are stored in replay buffers and sampled
during training.

This is different from downloading a fixed dataset before training.

------------------------------------------------------------------------

## 11. Reproducibility Notes

When using any external environment or repository, record:

-   Repository/project name.
-   URL.
-   Version or commit where practical.
-   Python version.
-   Gymnasium version.
-   PyTorch version.
-   Environment ID.
-   Observation preprocessing.
-   Action mapping.
-   Reward function.
-   Random seed.
-   Training configuration.
-   Any modifications made to the external environment.

This is particularly important for the Flappy Bird environment and
Atari/Pac-Man setup because the project's Milestone 3 configuration may
differ from an environment's default settings.

------------------------------------------------------------------------

## 12. Final Recommended Set

For the actual project work, the main links to keep easily accessible
are:

1.  **Gymnasium Documentation**
2.  **Gymnasium GitHub**
3.  **Gymnasium Atari Documentation**
4.  **ALE Pac-Man Documentation**
5.  **Flappy Bird Gymnasium**
6.  **Mnih et al. DQN Paper**
7.  **Dueling Network Architectures Paper**
8.  **ALE Paper**
9.  **Approved Pac-Man research paper**
10. **Approved Snake research paper**
11. **Approved Flappy Bird research paper**

Keep **Minari, D4RL, and CALE** only as optional background/future-work
resources, not as core project dependencies.
