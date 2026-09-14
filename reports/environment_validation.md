# Environment & Preprocessing Validation Report: Snake

**Status:** PASS  
**Target:** Snake Environment Skeleton, Visual Preprocessing Pipeline & Visual Inspection (Part 2, Steps 1–4)  
**Verification Date:** 2026-09-11  

---

## 1. Environment Specification

The Snake environment ([`SnakeEnv`](file:///E:/Technical_Projects/Cornerstone-project/src/environments/snake_env.py#L48-L260)) has been implemented following the specifications in [GUIDE-UPDATED.md](file:///E:/Technical_Projects/Cornerstone-project/GUIDE-UPDATED.md):

* **Game Map Dimensions:** $240 \times 240$ pixels
* **Logical Grid Dimensions:** $12 \times 12$ grid cells ($20\text{px} \times 20\text{px}$ per cell)
* **Initial Snake State:** Length = 3 cells, initial facing direction = `RIGHT`
* **Apple Spawning:** Exactly one apple present on the grid at any time, placed randomly on unoccupied cells
* **Growth Mechanics:** Consuming an apple increments the snake length by 1 and immediately triggers a new apple spawn
* **Termination Conditions:**
  1. Wall collision (head leaves $[0, 11]$ grid coordinate range)
  2. Self-collision (head collides with any existing body segment)
  3. Max steps limit (truncation)
* **Reward Structure (Configurable):**
  - Apple consumption: `+1.0`
  - Collision / Death: `-1.0`
  - Normal step / Survival: `0.0` (default)
* **Contract:** Standard Gymnasium `reset(seed=None, options=None)` and `step(action)`.

---

## 2. Base Observation Representations

1. **Grid Observation (`obs_type="grid"` - default):**
   - **Shape:** `(12, 12)`
   - **Type:** `np.int8` (values: `0` = Empty, `1` = Snake Head, `2` = Snake Body, `3` = Apple)
   - **Space:** `spaces.Box(low=0, high=3, shape=(12, 12), dtype=np.int8)`

2. **RGB Observation (`obs_type="rgb"`):**
   - **Shape:** `(240, 240, 3)`
   - **Type:** `np.uint8` with distinct RGB colors for background, snake head, snake body, and apple.
   - **Space:** `spaces.Box(low=0, high=255, shape=(240, 240, 3), dtype=np.uint8)`

---

## 3. Centralized Action Mapping

Action constants and direction vectors are centralized in [`SnakeAction`](file:///E:/Technical_Projects/Cornerstone-project/src/environments/snake_env.py#L21-L26):

| Action Enum | Integer | Name | Direction Vector $(\Delta x, \Delta y)$ | Opposite Action |
|:---:|:---:|:---:|:---:|:---:|
| `SnakeAction.UP` | `0` | `UP` | $(0, -1)$ | `DOWN` |
| `SnakeAction.DOWN` | `1` | `DOWN` | $(0, 1)$ | `UP` |
| `SnakeAction.LEFT` | `2` | `LEFT` | $(-1, 0)$ | `RIGHT` |
| `SnakeAction.RIGHT` | `3` | `RIGHT` | $(1, 0)$ | `LEFT` |

* **Reversal Handling:** When `allow_reverse=False` and snake length $> 1$, a directly opposite input action is safely ignored to maintain the current direction and prevent instant neck-collision suicide.

---

## 4. Visual Preprocessing & Four-Frame Stacking

> [!NOTE]
> This section validates the transformation of raw $240 \times 240 \times 3$ RGB observations into the $(4, 64, 64)$ DQN input tensor representation. DQN agent learning, neural networks, and replay buffers remain intentionally uninstantiated at this stage.

### 4.1 Visual Preprocessing Pipeline (`SnakePreprocessor`)
Implemented in [`src/preprocessing/snake_preprocess.py`](file:///E:/Technical_Projects/Cornerstone-project/src/preprocessing/snake_preprocess.py):

```
Raw RGB Frame (240, 240, 3)
         │
         ▼
Input Validation (3D, 3 Channels, non-empty)
         │
         ▼
Bilinear Spatial Downsampling (64 x 64 x 3)
         │
         ▼
RGB to HSV Transformation (H in [0, 360), S in [0, 1], V in [0, 1])
         │
         ▼
Background Suppression & Feature Extraction
         │
         ▼
Single-Channel Processed Frame (64, 64) [float32 in [0.0, 1.0]]
```

### 4.2 Implementation Choices & Configuration
- **HSV Thresholds:** The research guide specifies "HSV/background processing" but omits explicit numeric threshold constants. The following deterministic thresholds were selected:
  - `DEFAULT_BACKGROUND_VALUE_THRESHOLD = 0.10`
  - `DEFAULT_BACKGROUND_SATURATION_THRESHOLD = 0.05`
  - Pixels falling below both brightness and saturation cutoffs are mapped to background value $0.0$.
- **Output Representation & Numerical Range:**
  - Standard mode (`masked_value`): Retains normalized luminance/Value $V \in [0.0, 1.0]$ for foreground elements, setting background to $0.0$.
  - Data type: `np.float32` in range $[0.0, 1.0]$ (also supports `np.uint8` in $[0, 255]$ via configuration).

### 4.3 Temporal Frame Stacking (`FrameStack`)
Implemented in [`src/preprocessing/frame_stack.py`](file:///E:/Technical_Projects/Cornerstone-project/src/preprocessing/frame_stack.py):
- **Buffer Size:** Fixed window of 4 spatial frames.
- **Reset Behavior:** On environment reset, the initial processed frame is replicated across all 4 slots $(t_0, t_0, t_0, t_0)$ to establish the initial state without leaking temporal frames from prior episodes.
- **Sliding Window:** At each step $t$, the oldest frame $t-3$ is evicted and the new frame $t$ is appended, preserving strict temporal ordering:
  $$\text{Index } 0 \to t-3,\quad \text{Index } 1 \to t-2,\quad \text{Index } 2 \to t-1,\quad \text{Index } 3 \to t$$
- **Final Stacked Output Shape:** `(4, 64, 64)` of dtype `np.float32`.

---

## 5. Visual Preprocessing Validation

Visual qualitative inspection was conducted using [`src/visualization/visualize_snake_preprocessing.py`](file:///E:/Technical_Projects/Cornerstone-project/src/visualization/visualize_snake_preprocessing.py) with seed 42.

### 5.1 Visual Evidence Generated
All artifacts are persisted under [`observations/snake/`](file:///E:/Technical_Projects/Cornerstone-project/observations/snake/):

| File Path | Description | Shape / Representation |
|:---|:---|:---:|
| [`raw_initial.png`](file:///E:/Technical_Projects/Cornerstone-project/observations/snake/raw_initial.png) | Raw RGB initial observation from `reset()` | $(240, 240, 3)$ `uint8` |
| [`processed_initial.png`](file:///E:/Technical_Projects/Cornerstone-project/observations/snake/processed_initial.png) | Downsampled & background-filtered initial frame | $(64, 64)$ `float32` |
| [`processed_after_movement.png`](file:///E:/Technical_Projects/Cornerstone-project/observations/snake/processed_after_movement.png) | Processed frame following a normal step RIGHT | $(64, 64)$ `float32` |
| [`processed_near_apple.png`](file:///E:/Technical_Projects/Cornerstone-project/observations/snake/processed_near_apple.png) | Processed frame with snake head positioned adjacent to apple | $(64, 64)$ `float32` |
| [`processed_apple_eaten.png`](file:///E:/Technical_Projects/Cornerstone-project/observations/snake/processed_apple_eaten.png) | Processed frame immediately upon eating the apple and growing | $(64, 64)$ `float32` |
| [`processed_collision.png`](file:///E:/Technical_Projects/Cornerstone-project/observations/snake/processed_collision.png) | Processed frame at terminal boundary wall collision | $(64, 64)$ `float32` |
| [`comparison_initial.png`](file:///E:/Technical_Projects/Cornerstone-project/observations/snake/comparison_initial.png) | Side-by-side Raw RGB vs Processed Frame comparison with colorbar | Figure |
| [`frame_stack.png`](file:///E:/Technical_Projects/Cornerstone-project/observations/snake/frame_stack.png) | 4-channel temporal stack progression across $t-3$ to $t$ | Figure (4 channels) |

### 5.2 Inspection Findings
- **Background Suppression:** The dark grid background ($[15, 15, 15]$) is cleanly suppressed to $0.0$ across all frames.
- **Entity Distinguishability:**
  - **Apple:** Apple occupies a prominent high-intensity region ($\sim 1.0$) distinct from empty grid space.
  - **Snake Head:** Rendered with highest value intensity ($1.0$), making head direction and positioning clear.
  - **Snake Body:** Body segments maintain clear intermediate intensity ($\sim 0.71$) and contiguous spatial connectivity.
- **Spatial Resolution & Artifacts:** Resizing to $64 \times 64$ preserves entity aspect ratios and positions without blurring out single-cell elements.
- **Temporal Progression:** In `frame_stack.png`, Channel 0 ($t-3$) through Channel 3 ($t$) visibly track head advancement, turning upward, apple consumption, and tail expansion.
- **Limitations:** Background grid lines are omitted in the preprocessed frame by design of the background filter; only active gameplay entities are highlighted to minimize visual noise for the DQN agent.

---

## 6. Runtime Evidence & Logs

1. **Environment Smoke Log:** [`logs/snake_environment_smoke_test.csv`](file:///E:/Technical_Projects/Cornerstone-project/logs/snake_environment_smoke_test.csv) (126 recorded transitions).
2. **Preprocessing Pipeline Smoke Log:** [`logs/snake_preprocessing_smoke_test.csv`](file:///E:/Technical_Projects/Cornerstone-project/logs/snake_preprocessing_smoke_test.csv) (33 recorded multi-step transitions).

---

## 7. Verification Summary

* **Test Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/ -v
  ```
* **Total Tests Executed:** 40
* **Total Tests Passed:** 40
* **Total Tests Failed:** 0
* **Overall Status:** **PASS**
