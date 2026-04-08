# TurtleBot3 Machine Learning

A ROS 2 reinforcement learning framework for autonomous TurtleBot3 navigation using **PPO** (Proximal Policy Optimization) with a continuous action space and **DQN** (Deep Q-Network) with a discrete action space. Both agents learn to navigate to goals while avoiding obstacles in Gazebo simulation.

**Supported ROS 2 distributions:** Humble · Jazzy · Rolling

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Packages](#packages)
  - [turtlebot3\_ppo](#turtlebot3_ppo)
  - [turtlebot3\_dqn](#turtlebot3_dqn)
- [ROS 2 Parameters](#ros-2-parameters)
- [Training Stages](#training-stages)
- [Saved Models](#saved-models)
- [Visualization](#visualization)
- [License](#license)

---

## Overview

| Feature | PPO | DQN |
|---|---|---|
| Algorithm | On-policy policy gradient | Off-policy value-based |
| Action space | Continuous 2D | Discrete (5 actions) |
| Linear velocity | Learned [0.0, 0.22] m/s | Fixed per action |
| Angular velocity | Learned [−2.84, 2.84] rad/s | {±1.5, ±0.75, 0.0} rad/s |
| State dimension | 50 (distance, heading, 48 LiDAR) | 50 |
| Experience | On-policy rollout buffer | Prioritized replay (500 K) |
| Exploration | Learned Gaussian policy | ε-greedy decay |
| Model format | PyTorch `.pt` | Keras `.h5` + `.json` |

The **state vector** is identical for both agents:

```
[goal_distance, goal_angle, scan[0], scan[1], ..., scan[47]]   # dim = 50
```

**Terminal conditions:**
- Success — `reward == +100.0` (robot within 0.20 m of goal)
- Collision — `reward == −50.0` (any LiDAR reading < 0.15 m or timeout)

---

## Architecture

Each algorithm runs as three cooperating ROS 2 nodes:

```
┌─────────────────────┐        service calls         ┌──────────────────────┐
│     ppo_agent        │ ◄────────────────────────── │   rl_environment      │
│  (training loop)     │ ────────────────────────── ► │  (reward + state)    │
└─────────────────────┘                               └──────────┬───────────┘
                                                                  │ service calls
                                                       ┌──────────▼───────────┐
                                                       │  ppo_gazebo           │
                                                       │  (goal spawning)      │
                                                       └──────────────────────┘
```

**PPO node communication:**

| Interface | Type | Direction | Purpose |
|---|---|---|---|
| `rl_agent_interface` | `Ppo` service | agent → env | Send action, receive (state, reward, done) |
| `make_environment` | `Empty` service | agent → env | Initialize simulation |
| `reset_environment` | `Ppo` service | agent → env | Reset episode |
| `ppo_initialize_env` | `Goal` service | env → gazebo | Spawn initial goal |
| `ppo_task_succeed` | `Goal` service | env → gazebo | Reposition goal on success |
| `ppo_task_failed` | `Goal` service | env → gazebo | Reset robot on failure |
| `/ppo_action` | `Float32MultiArray` | agent → any | [linear\_vel, angular\_vel, ep\_reward, step\_reward] |
| `/ppo_result` | `Float32MultiArray` | agent → graph | [ep\_reward, policy\_loss, value\_loss, entropy] |
| `/odom` | `Odometry` | gazebo → env | Robot pose |
| `/scan` | `LaserScan` | gazebo → env | 360° LiDAR |
| `/clock` | `Clock` | gazebo → env | Sim-time synchronisation |
| `cmd_vel` | `Twist` / `TwistStamped` | env → gazebo | Velocity command |

> The DQN ecosystem uses the same topology with `Dqn`-typed services and unprefixed Gazebo service names (`initialize_env`, `task_succeed`, `task_failed`).

---

## Requirements

- ROS 2 Humble, Jazzy, or Rolling
- Gazebo (Classic or Ignition, depending on distro)
- Python ≥ 3.10
- PyTorch ≥ 2.0 (CPU or CUDA)
- `turtlebot3_msgs` — custom `Ppo`, `Dqn`, and `Goal` service types

Python dependencies are declared in each package's `setup.py` and installed automatically by `pip` during the build.

---

## Installation

```bash
# 1. Create or enter your ROS 2 workspace
mkdir -p ~/turtlebot3_ws/src && cd ~/turtlebot3_ws/src

# 2. Clone this repository
git clone https://github.com/praxi027/turtlebot3_machine_learning.git

# 3. Install turtlebot3_msgs (required for service definitions)
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git

# 4. Build
cd ~/turtlebot3_ws
colcon build --symlink-install

# 5. Source the workspace
source install/setup.bash
```

Set your robot model before launching:

```bash
export TURTLEBOT3_MODEL=burger   # or waffle / waffle_pi
```

---

## Quick Start

### PPO Training

Open three terminals (all sourced):

```bash
# Terminal 1 — Gazebo simulation (stage 1: empty arena)
ros2 run turtlebot3_ppo ppo_gazebo 1

# Terminal 2 — RL environment
ros2 run turtlebot3_ppo ppo_environment

# Terminal 3 — PPO agent
ros2 run turtlebot3_ppo ppo_agent
```

Override parameters inline:

```bash
ros2 run turtlebot3_ppo ppo_agent \
  --ros-args \
  -p max_training_episodes:=2000 \
  -p rollout_steps:=4096 \
  -p entropy_coeff:=0.05 \
  -p experiment_name:=my_run
```

Or use a YAML params file (ROS 2 params-file format):

```bash
ros2 run turtlebot3_ppo ppo_agent \
  --ros-args --params-file my_experiment.yaml
```

Resume from a checkpoint:

```bash
ros2 run turtlebot3_ppo ppo_agent \
  --ros-args \
  -p model_file:=model100.pt \
  -p experiment_name:=my_run
```

### DQN Training

```bash
# Terminal 1
ros2 run turtlebot3_dqn dqn_gazebo 1

# Terminal 2
ros2 run turtlebot3_dqn dqn_environment

# Terminal 3
ros2 run turtlebot3_dqn dqn_agent
```

### DQN Inference (no training)

```bash
ros2 run turtlebot3_dqn dqn_test \
  --ros-args -p model_file:=saved_model/model1.h5
```

---

## Packages

### turtlebot3_ppo

**`ppo_agent.py`** — Main training node.

- **Network** (`ActorCritic`): shared backbone (50 → 512 → 256 → 128, ReLU) with separate actor (mean + learnable log\_std, tanh-squashed) and critic (scalar value) heads.
- **Buffer** (`RolloutBuffer`): stores states, actions, pre-tanh actions, log-probs, rewards, dones, and values for one full rollout.
- **Update**: Generalized Advantage Estimation (GAE) followed by PPO clipped surrogate loss with mini-batch gradient descent.
- **Action smoothing**: optional EMA blending (`action_smoothing` α) between the previous and current action to reduce jitter.
- **Logging**: TensorBoard metrics — episode reward, policy/value/entropy loss, success/collision rate, explained variance, action statistics.

**`ppo_environment.py`** — Reward and state computation node.

Reward at each nonterminal step:

```
r = progress_scale × (d_prev − d_curr)          # goal progress
  + yaw_scale × (1 − 2|θ|/π)                    # heading alignment
  + obstacle_scale × clip(obstacle_penalty, 0, 1) # safety (quadratic)
  + lyapunov_scale × (γ·Φ(s') − Φ(s))           # potential shaping (optional)
```

Terminal: `+100.0` on success, `−50.0` on collision, timeout, or danger-zone entry.

**`ppo_gazebo.py`** — Gazebo interface for goal spawning and robot resets. Generates random goal positions within [−2.1, 2.1] m (stages 1–3), cycles through a fixed list (stage 4), and also supports named custom scenarios such as `warehouse_easy` with a fixed goal and full episode reset on success. Detects ROS 2 distro and uses the appropriate Gazebo API.

**`result_graph.py`** — PyQt5 + PyQtGraph live dashboard showing episode reward, policy loss, and value loss.

---

### turtlebot3_dqn

**`dqn_agent.py`** — DQN with Prioritized Experience Replay (PER, α=0.6, β=0.4). Target network updated every 5 000 steps. ε decays linearly to 0.05 over `epsilon_decay` steps.

**`dqn_environment.py`** — Same 50D state as PPO; maps the 5 discrete actions to angular velocity commands.

**`dqn_gazebo.py`** — Equivalent to `ppo_gazebo.py` but uses DQN-specific service names.

**`dqn_test.py`** — Loads a saved model and runs greedy inference (no exploration).

**`action_graph.py`** — Bar chart showing Q-value per action in real time.

**`result_graph.py`** — Episode reward and average max Q-value plots.

---

## ROS 2 Parameters

### PPO Agent (`ppo_agent`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_training_episodes` | int | 1000 | Training episode limit |
| `model_file` | string | `''` | Checkpoint filename to resume from |
| `experiment_name` | string | `''` | Logical experiment name used in the run path |
| `run_id` | string | `''` | Optional run identifier; autogenerated when unset |
| `use_gpu` | bool | `true` | Enable CUDA |
| `verbose` | bool | `true` | Per-step console output |
| `logging` | bool | `true` | TensorBoard logging |
| `tensorboard_dir` | string | `''` | Explicit TensorBoard output directory |
| `checkpoint_dir` | string | `''` | Explicit checkpoint output directory |
| `gamma` | double | 0.99 | Discount factor |
| `gae_lambda` | double | 0.95 | GAE smoothing (λ) |
| `clip_epsilon` | double | 0.2 | PPO clip range |
| `learning_rate` | double | 3e-4 | Adam LR |
| `rollout_steps` | int | 2048 | Trajectory length before update |
| `n_epochs` | int | 10 | Policy update epochs per rollout |
| `minibatch_size` | int | 64 | Mini-batch size |
| `value_coeff` | double | 0.5 | Value loss weight |
| `entropy_coeff` | double | 0.01 | Entropy bonus weight |
| `max_grad_norm` | double | 0.5 | Gradient clipping norm |
| `save_interval` | int | 100 | Episodes between checkpoint saves |
| `angular_vel_max` | double | 2.84 | Maximum angular velocity (rad/s) |
| `action_smoothing` | double | 0.0 | EMA smoothing factor α ∈ [0, 1] |

### PPO Environment (`rl_environment`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reward_progress_scale` | double | 5.0 | Goal-progress reward coefficient |
| `reward_yaw_scale` | double | 0.5 | Heading alignment reward coefficient |
| `reward_obstacle_scale` | double | −5.0 | Obstacle penalty amplitude |
| `reward_obstacle_safe_dist` | double | 0.6 | Distance (m) at which penalty is zero |
| `reward_obstacle_danger_dist` | double | 0.15 | Distance (m) at which penalty is maximum |
| `reward_success` | double | 100.0 | Terminal reward for reaching goal |
| `reward_fail` | double | −50.0 | Terminal reward for collision, timeout, or danger-zone entry |
| `danger_zones` | string[] | `['']` | Keep-out regions encoded as `"x_min,y_min,x_max,y_max"` for boxes or `"circle,x,y,radius"` for circular zones |
| `max_step` | int | 800 | Episode timeout (steps) |
| `goal_threshold` | double | 0.20 | Goal-reached distance (m) |
| `collision_threshold` | double | 0.15 | Collision distance (m) |
| `angular_vel_max` | double | 2.84 | Angular velocity clip (rad/s) |
| `lyapunov_scale` | double | 0.0 | Lyapunov/potential-based shaping coefficient |

### DQN Agent (`dqn_agent`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_training_episodes` | int | 1000 | Training episode limit |
| `epsilon_decay` | int | 6000 | Steps for ε-greedy annealing |
| `model_file` | string | `''` | Checkpoint file to resume from |
| `use_gpu` | bool | `false` | Enable CUDA |
| `verbose` | bool | `true` | Console output |

---

## Training Stages

The Gazebo environment supports four stages of increasing difficulty:

| Stage | Description |
|---|---|
| 1 | Empty arena — no static obstacles |
| 2 | Static obstacles, open space |
| 3 | Dense static obstacles |
| 4 | Fixed set of goal positions for reproducible evaluation |

Pass the stage number as a positional argument to the Gazebo node:

```bash
ros2 run turtlebot3_ppo ppo_gazebo 2
```

Custom named scenarios can also be launched through the workspace helper scripts. The repository currently includes `warehouse_easy`, a single-rack warehouse map with one unsafe upper aisle, a wider safe lower detour, and a visible pickup marker at the goal.

```bash
./start_experiment.sh experiments/scenarios/warehouse_easy.yaml 10 warehouse_easy
```

---

## Saved Models

Workspace-launched PPO checkpoints are stored under `~/turtlebot3_runs/`.
Direct `ppo_agent` runs also default there under `manual/`.

Legacy checkpoints may still exist under `turtlebot3_ppo/saved_model/`.

| Directory | Description |
|---|---|
| *(root)* | Sequential training run (models 1–21) |
| `baseline/` | Reference configuration |
| `bigger_rollout/` | `rollout_steps = 4096` |
| `high_entropy/` | `entropy_coeff = 0.05` |
| `lower_angular/` | Reduced angular velocity bound |
| `smaller_lr/` | `learning_rate = 1e-4` |
| `more_episodes/` | Extended training |
| `stage2_action_smoothing/` | `action_smoothing = 0.2`, stage 2 |
| `stage2_baseline/` | Stage 2 reference |

Load a checkpoint:

```bash
ros2 run turtlebot3_ppo ppo_agent \
  --ros-args \
  -p model_file:=model100.pt \
  -p experiment_name:=baseline
```

---

## Visualization

**Live training dashboard (PPO):**

```bash
ros2 run turtlebot3_ppo result_graph
```

Displays episode reward, policy loss, and value loss in real time.

**TensorBoard:**

```bash
tensorboard --logdir ~/turtlebot3_runs/
```

PPO runs are written to:

```text
~/turtlebot3_runs/<stage-or-manual>/<experiment>/<run_id>/
```

**DQN action viewer:**

```bash
ros2 run turtlebot3_dqn action_graph
```

**DQN result graph:**

```bash
ros2 run turtlebot3_dqn result_graph
```

---

## License

Distributed under the [Apache 2.0 License](LICENSE).
