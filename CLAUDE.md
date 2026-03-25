# Snake RL — CLAUDE.md

## What this project is

Interactive Reinforcement Learning demo: a snake learns to play Snake using a DQN (Deep Q-Network) agent. Training runs in Python and the visualization happens in real time in the browser via WebSocket.

## How to run

```bash
pip install -r requirements.txt
python snake_server.py
# Open http://localhost:5000
```

If PyTorch is not available, the server starts anyway with a random agent (useful for testing the interface).

## File structure

```
snake_server.py   — Flask server + game logic + DQN model
index.html        — frontend (canvas, charts, controls)
requirements.txt  — Python dependencies
CLAUDE.md         — this file
```

## Architecture

### Backend (`snake_server.py`)

**`SnakeGame`** — game logic isolated from the model.
- `reset()` → resets and returns the initial state (11-float vector)
- `step(action)` → applies action, returns `(next_state, reward, done)`
- `get_render_data()` → dict with snake positions, food, and score for the frontend
- State has 11 dimensions: danger ahead/right/left, current direction (4 bits), relative food position (4 bits)
- Rewards: `+10` for eating, `-10` for dying, `+1/-1` for moving closer/further from food

**`DQN` (nn.Module)** — network 11 → 128 → 128 → 4 (Linear + ReLU).

**`Agent`** — wraps the DQN with experience replay and ε-greedy policy.
- `act(state)` → returns action (exploration or exploitation)
- `remember(...)` → saves transition to replay buffer (deque of 10k)
- `train_step()` → samples batch of 64 and runs backprop
- `update_target()` → copies weights to target network (every 10 episodes)
- `reset()` → resets weights, buffer and epsilon

**Training loop** (`training_loop`) — runs in a separate thread. Each step: act → remember → train → emit frame via WebSocket (~15fps with `time.sleep(0.065)`).

**WebSocket events (server → client):**
- `frame` — current game state at each step (positions + epsilon + episode)
- `episode_end` — episode metrics (score, 50-ep average, best, history)
- `status` — state change (running/paused/reset)

**WebSocket events (client → server):**
- `start` — starts the training thread
- `stop` — sets `running = False` (thread stops on next cycle)
- `reset` — stops, resets agent and history, emits initial state

### Frontend (`index.html`)

- 400×400px canvas, 20×20 grid, 20px cell size
- Chart.js for score-per-episode chart with 50-episode moving average
- Socket.IO to receive frames and emit commands
- Epsilon (exploration) progress bar updates every frame

## Hyperparameters

| Constant | Value | Description |
|---|---|---|
| `GRID_W/H` | 20 | Grid size |
| `STATE_SIZE` | 11 | State vector dimension |
| `ACTION_SIZE` | 4 | Number of actions (straight, right, left, straight) |
| `BATCH_SIZE` | 64 | Training batch size |
| `GAMMA` | 0.95 | Discount factor |
| `LR` | 0.001 | Learning rate (Adam) |
| `EPSILON_START` | 1.0 | Initial exploration (100%) |
| `EPSILON_MIN` | 0.01 | Minimum exploration (1%) |
| `EPSILON_DECAY` | 0.995 | Decay per episode |
| `MEMORY_SIZE` | 10,000 | Replay buffer capacity |
| `TARGET_UPDATE` | 10 | Episodes between target network syncs |

## Common extension points

**Save and load the model:**
```python
torch.save(agent.model.state_dict(), "model.pth")
agent.model.load_state_dict(torch.load("model.pth"))
```
Add `save` and `load` WebSocket endpoints and buttons in the frontend.

**Speed up training without rendering:**
Reduce `time.sleep(0.065)` to `0.01` or remove the sleep and emit frames only every N steps.

**Adjust learning speed:**
Reduce `EPSILON_DECAY` (e.g. `0.99`) so the agent explores more before converging. Increasing `LR` speeds up learning but may destabilise training.

**Expand the state:**
`_get_state()` in `SnakeGame` is the place to add more features (e.g. distance to own body, ray-cast vision).

## Known issues

- `action == 3` and `action == 0` have the same effect (go straight). The action space has 4 outputs for symmetry, but there are effectively only 3 distinct behaviours.
- The training thread uses `global running` — not thread-safe for multiple simultaneous connections. For production use, add locks or rewrite with `asyncio`.
- `time.sleep(0.065)` blocks the thread and may delay Socket.IO on slow machines.
