# Snake RL — Deep Q-Network Snake Game

An interactive reinforcement learning demo where a DQN agent learns to play Snake in real time, visualized through a web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-optional-orange) ![Flask](https://img.shields.io/badge/Flask-web--server-lightgrey)

---

## Overview

The agent uses a **Deep Q-Network (DQN)** with experience replay and a target network to learn the Snake game from scratch. You can watch it train in real time via a browser-based dashboard showing the game canvas, score history chart, and training metrics.

---

## Features

- Real-time Snake gameplay rendered in the browser (Canvas API)
- DQN agent with epsilon-greedy exploration and experience replay
- Live metrics: current score, 50-episode moving average, best score, epsilon
- Score history chart (Chart.js)
- Start / Stop / Reset controls
- Graceful fallback to a random agent if PyTorch is not installed

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, Flask-SocketIO |
| Deep Learning | PyTorch (optional) |
| Numerical Computing | NumPy |
| Frontend | HTML5 Canvas, JavaScript, Socket.IO, Chart.js 4 |

---

## Installation

```bash
# Clone or download the repository, then:
cd snake-rl
pip install -r requirements.txt
```

> PyTorch is listed as a dependency. If you skip it, the server runs with a random agent (useful for UI testing only).

---

## Running

```bash
python snake_server.py
```

Open `http://localhost:5000` in your browser and click **Start**.

---

## Project Structure

```
snake-rl/
├── snake_server.py   # Flask server + DQN agent + Snake game logic
├── index.html        # Web UI (served by Flask)
├── requirements.txt  # Python dependencies
└── CLAUDE.md         # Technical architecture notes
```

---

## Architecture

### Backend (`snake_server.py`)

**`SnakeGame`** — game engine on a 20×20 grid.
Produces an 11-dimensional state vector per step:
- 3 danger flags (ahead / relative right / relative left)
- 4 one-hot direction encoding
- 4 food direction flags (left / right / up / down)

**`DQN`** — two-layer MLP (`11 → 128 → 128 → 4`) implemented in PyTorch.
Two instances: main network (trained each step) and target network (synced every 10 episodes for stability).

**`Agent`** — ε-greedy policy, experience replay buffer (capacity 10 000), Adam optimizer.

**Reward shaping:**

| Event | Reward |
|-------|--------|
| Eating food | +10 |
| Dying | −10 |
| Moving closer to food | +1 |
| Moving away from food | −1 |

The training loop runs in a background daemon thread and emits ~15 FPS game frames to the frontend via WebSocket.

### Frontend (`index.html`)

- 400×400 px canvas, 20 px cell size
- Snake head rendered with rounded corners; body fades toward the tail
- Score history chart updates at episode end (not per frame)
- Scrollable log of the last 200 episodes

### WebSocket Events

| Direction | Event | Description |
|-----------|-------|-------------|
| Client → Server | `start` | Begin training |
| Client → Server | `stop` | Pause training |
| Client → Server | `reset` | Reset agent and metrics |
| Server → Client | `frame` | Game state at ~15 FPS |
| Server → Client | `episode_end` | End-of-episode metrics |
| Server → Client | `status` | Running / paused state |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GRID_W / GRID_H` | 20 | World dimensions |
| `STATE_SIZE` | 11 | Network input size |
| `ACTION_SIZE` | 4 | Network output size |
| `BATCH_SIZE` | 64 | Training batch size |
| `GAMMA` | 0.95 | Discount factor |
| `LR` | 0.001 | Learning rate (Adam) |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_MIN` | 0.01 | Minimum exploration rate |
| `EPSILON_DECAY` | 0.995 | Per-episode epsilon decay |
| `MEMORY_SIZE` | 10 000 | Replay buffer capacity |
| `TARGET_UPDATE` | 10 | Target network sync interval (episodes) |
| `MAX_STEPS` | 400 | Max steps per episode |
| `FPS_DELAY` | 0.065 s | Frame delay (~15 FPS) |

---

## Known Limitations

- Actions 0 and 3 currently have the same effect (redundant action in the space).
- Global `running` flag is not thread-safe for multiple simultaneous browser connections.
- No model persistence — trained weights are lost when the server stops.

---

## License

This project is for personal and educational use.
