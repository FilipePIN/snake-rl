import os
import random
import json
import time
import threading
import numpy as np
from collections import deque
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Settings ---
GRID_SIZE = 20
GRID_W = 20
GRID_H = 20
STATE_SIZE = 11       # simplified state vector
ACTION_SIZE = 4       # up, down, left, right
BATCH_SIZE = 64
GAMMA = 0.95
LR = 0.003
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.990
MEMORY_SIZE = 10_000
TARGET_UPDATE = 20    # episodes between target network updates

# --- Directions ---
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRS = [UP, RIGHT, DOWN, LEFT]

# --- Snake Game ---
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        cx, cy = GRID_W // 2, GRID_H // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = RIGHT
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.max_steps = GRID_W * GRID_H * 2
        return self._get_state()

    def _place_food(self):
        while True:
            f = (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
            if f not in self.snake:
                return f

    def _get_state(self):
        head = self.snake[0]
        d = self.direction

        # Dangers: straight, relative right, relative left
        def danger(dir_):
            nx, ny = head[0] + dir_[0], head[1] + dir_[1]
            return int(
                nx < 0 or nx >= GRID_W or
                ny < 0 or ny >= GRID_H or
                (nx, ny) in self.snake[1:]
            )

        idx = DIRS.index(d)
        right_dir = DIRS[(idx + 1) % 4]
        left_dir  = DIRS[(idx - 1) % 4]

        fx, fy = self.food
        hx, hy = head

        state = [
            danger(d),
            danger(right_dir),
            danger(left_dir),
            int(d == LEFT), int(d == RIGHT),
            int(d == UP),   int(d == DOWN),
            int(fx < hx), int(fx > hx),
            int(fy < hy), int(fy > hy),
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        old_dir_idx = DIRS.index(self.direction)

        if action == 0:   # straight
            pass
        elif action == 1: # turn right
            self.direction = DIRS[(old_dir_idx + 1) % 4]
        elif action == 2: # turn left
            self.direction = DIRS[(old_dir_idx - 1) % 4]
        # action == 3 → straight too (redundant, keeps 4 symmetric outputs)

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Death
        if (
            new_head[0] < 0 or new_head[0] >= GRID_W or
            new_head[1] < 0 or new_head[1] >= GRID_H or
            new_head in self.snake[1:] or
            self.steps > self.max_steps
        ):
            return self._get_state(), -10, True

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 10
        else:
            self.snake.pop()
            # Reward for moving closer to food
            dist_before = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            dist_after  = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 1 if dist_after < dist_before else -1

        return self._get_state(), reward, False

    def get_render_data(self):
        return {
            "snake": list(self.snake),
            "food": list(self.food),
            "score": self.score,
            "grid_w": GRID_W,
            "grid_h": GRID_H,
        }


# --- DQN with PyTorch ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class DQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_SIZE, 128), nn.ReLU(),
                nn.Linear(128, 128),        nn.ReLU(),
                nn.Linear(128, ACTION_SIZE)
            )
        def forward(self, x):
            return self.net(x)

    class Agent:
        def __init__(self):
            self.model  = DQN()
            self.target = DQN()
            self.target.load_state_dict(self.model.state_dict())
            self.opt    = optim.Adam(self.model.parameters(), lr=LR)
            self.memory = deque(maxlen=MEMORY_SIZE)
            self.epsilon = EPSILON_START
            self.episode = 0

        def act(self, state):
            if random.random() < self.epsilon:
                return random.randint(0, ACTION_SIZE - 1)
            with torch.no_grad():
                q = self.model(torch.FloatTensor(state))
            return q.argmax().item()

        def remember(self, s, a, r, s2, done):
            self.memory.append((s, a, r, s2, done))

        def train_step(self):
            if len(self.memory) < BATCH_SIZE:
                return None
            batch = random.sample(self.memory, BATCH_SIZE)
            s, a, r, s2, d = zip(*batch)
            S  = torch.FloatTensor(np.array(s))
            A  = torch.LongTensor(a).unsqueeze(1)
            R  = torch.FloatTensor(r)
            S2 = torch.FloatTensor(np.array(s2))
            D  = torch.FloatTensor(d)

            q_cur = self.model(S).gather(1, A).squeeze()
            with torch.no_grad():
                q_next = self.target(S2).max(1)[0]
            q_target = R + GAMMA * q_next * (1 - D)

            loss = nn.MSELoss()(q_cur, q_target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return loss.item()

        def update_epsilon(self):
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        def update_target(self):
            self.target.load_state_dict(self.model.state_dict())

        def reset(self):
            self.model  = DQN()
            self.target = DQN()
            self.target.load_state_dict(self.model.state_dict())
            self.opt    = optim.Adam(self.model.parameters(), lr=LR)
            self.memory.clear()
            self.epsilon = EPSILON_START
            self.episode = 0

    TORCH_OK = True

except ImportError:
    TORCH_OK = False
    print("[WARNING] PyTorch not found. Using random agent for demo.")

    class Agent:
        def __init__(self):
            self.epsilon = 1.0
            self.episode = 0
        def act(self, state):
            return random.randint(0, 3)
        def remember(self, *args): pass
        def train_step(self): return None
        def update_epsilon(self):
            self.epsilon = max(0.01, self.epsilon * 0.995)
        def update_target(self): pass
        def reset(self):
            self.epsilon = 1.0
            self.episode = 0


# --- Flask + SocketIO ---
app = Flask(__name__)
app.config["SECRET_KEY"] = "snakerl"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

game  = SnakeGame()
agent = Agent()
scores_history = []
running = False
training_thread = None

def training_loop():
    global running
    while running:
        state = game.reset()
        done  = False
        total_reward = 0

        while not done and running:
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

            # Emit frame at ~15fps
            socketio.emit("frame", {
                **game.get_render_data(),
                "epsilon": round(agent.epsilon, 3),
                "episode": agent.episode,
            })
            time.sleep(0.065)

        if running:
            agent.update_epsilon()
            agent.episode += 1
            scores_history.append(game.score)
            avg = round(sum(scores_history[-50:]) / min(len(scores_history), 50), 2)

            if agent.episode % TARGET_UPDATE == 0:
                agent.update_target()

            socketio.emit("episode_end", {
                "episode":  agent.episode,
                "score":    game.score,
                "avg_score": avg,
                "epsilon":  round(agent.epsilon, 3),
                "best":     max(scores_history),
                "history":  scores_history[-200:],
            })


@socketio.on("connect")
def on_connect():
    emit("status", {"running": running, "episode": agent.episode})

@socketio.on("start")
def on_start():
    global running, training_thread
    if not running:
        running = True
        training_thread = threading.Thread(target=training_loop, daemon=True)
        training_thread.start()
        emit("status", {"running": True})

@socketio.on("stop")
def on_stop():
    global running
    running = False
    emit("status", {"running": False})

@socketio.on("reset")
def on_reset():
    global running, scores_history
    running = False
    time.sleep(0.2)
    agent.reset()
    scores_history = []
    game.reset()
    emit("status", {"running": False, "episode": 0})
    emit("episode_end", {
        "episode": 0, "score": 0, "avg_score": 0,
        "epsilon": 1.0, "best": 0, "history": []
    })


HTML_PAGE = open("index.html").read() if __name__ != "__main__" else ""

@app.route("/")
def index():
    try:
        with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<h1>Error loading index.html: {e}</h1><p>BASE_DIR: {BASE_DIR}</p>"

if __name__ == "__main__":
    print("=" * 50)
    print(f"  Snake RL Demo  |  PyTorch: {'OK' if TORCH_OK else 'NOT found'}")
    print(f"  Open: http://localhost:5000")
    print("=" * 50)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
