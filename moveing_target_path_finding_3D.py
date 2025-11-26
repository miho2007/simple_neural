import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# -----------------------------
# Agent class
# -----------------------------
class Agent:
    def __init__(self, start_pos=(0,0,0), hidden_size=8, learning_rate=0.05):
        self.pos = np.array(start_pos, dtype=float)
        self.positions = [self.pos.copy()]
        self.learning_rate = learning_rate

        input_size = 6  # x, y, z, tx, ty, tz
        output_size = 3  # dx, dy, dz

        # Tiny neural network
        self.w1 = np.random.uniform(-1,1,(hidden_size, input_size))
        self.b1 = np.random.uniform(-1,1,hidden_size)
        self.w2 = np.random.uniform(-1,1,(output_size, hidden_size))
        self.b2 = np.random.uniform(-1,1,output_size)

    def forward(self, input_vec):
        z1 = self.w1 @ input_vec + self.b1
        h = np.tanh(z1)
        z2 = self.w2 @ h + self.b2
        return h, z2

    def train_step(self, target_pos, iterations=3):
        tx, ty, tz = target_pos
        x, y, z = self.pos
        for _ in range(iterations):
            input_vec = np.array([x, y, z, tx, ty, tz])
            h, output = self.forward(input_vec)
            dx, dy, dz = output
            x_new, y_new, z_new = x + dx, y + dy, z + dz

            # Loss
            loss = (x_new - tx)**2 + (y_new - ty)**2 + (z_new - tz)**2

            # Backprop
            dL_dd = np.array([2*(x_new - tx), 2*(y_new - ty), 2*(z_new - tz)])
            dL_dw2 = np.outer(dL_dd, h)
            dL_db2 = dL_dd.copy()
            dL_dh = self.w2.T @ dL_dd
            dL_dz1 = dL_dh * (1 - h**2)
            dL_dw1 = np.outer(dL_dz1, input_vec)
            dL_db1 = dL_dz1.copy()

            # Update weights
            self.w2 -= self.learning_rate * dL_dw2
            self.b2 -= self.learning_rate * dL_db2
            self.w1 -= self.learning_rate * dL_dw1
            self.b1 -= self.learning_rate * dL_db1

    def move(self, target_pos, max_step=0.25):
        _, output = self.forward(np.array([*self.pos, *target_pos]))
        dx, dy, dz = output
        step_size = np.linalg.norm([dx, dy, dz])
        if step_size > max_step:
            dx *= max_step / step_size
            dy *= max_step / step_size
            dz *= max_step / step_size
        self.pos += np.array([dx, dy, dz])
        self.positions.append(self.pos.copy())

# -----------------------------
# Target class
# -----------------------------
class Target:
    def __init__(self, start_pos, speed=0.15):
        self.pos = np.array(start_pos, dtype=float)
        direction = np.array([0.0,0.0,0.0]) - self.pos
        direction /= np.linalg.norm(direction)
        self.velocity = direction * speed
        self.positions = [self.pos.copy()]

    def move(self):
        self.pos += self.velocity
        self.positions.append(self.pos.copy())

# -----------------------------
# Simulation parameters
# -----------------------------
max_steps = 120
max_step_size = 0.25

# Agents starting positions
agents = [
    Agent(start_pos=(0,0,0)),
    Agent(start_pos=(1,0,0)),
    Agent(start_pos=(0,1,0))
]

# Multiple targets starting positions
targets = [
    Target(start_pos=(12,15,8), speed=0.15),
    Target(start_pos=(10,14,12), speed=0.12),
    Target(start_pos=(14,10,10), speed=0.18)
   
]

# -----------------------------
# Simulation loop
# -----------------------------
for step in range(max_steps):
    # Move targets
    for target in targets:
        target.move()
    # Each agent chases the corresponding target
    for i, agent in enumerate(agents):
        target_pos = targets[i].positions[-1]
        agent.train_step(target_pos, iterations=3)
        agent.move(target_pos, max_step=max_step_size)

# -----------------------------
# Plot animation
# -----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2,15)
ax.set_ylim(-2,15)
ax.set_zlim(-2,15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

agent_colors = ['b','g','m']
target_colors = ['c','y','k']
agent_lines = []
agent_points = []
target_points = []

for i, agent in enumerate(agents):
    line, = ax.plot([],[],[], color=agent_colors[i], linewidth=2)
    point, = ax.plot([],[],[], 'o', color=agent_colors[i])
    agent_lines.append(line)
    agent_points.append(point)

for i, target in enumerate(targets):
    point, = ax.plot([target.positions[0][0]], [target.positions[0][1]], [target.positions[0][2]], 'o', color=target_colors[i])
    target_points.append(point)

def update(frame):
    for i, agent in enumerate(agents):
        xs = [p[0] for p in agent.positions[:frame+1]]
        ys = [p[1] for p in agent.positions[:frame+1]]
        zs = [p[2] for p in agent.positions[:frame+1]]
        agent_lines[i].set_data(xs, ys)
        agent_lines[i].set_3d_properties(zs)
        agent_points[i].set_data([xs[-1]], [ys[-1]])
        agent_points[i].set_3d_properties([zs[-1]])

    for i, target in enumerate(targets):
        tx, ty, tz = target.positions[frame]
        target_points[i].set_data([tx], [ty])
        target_points[i].set_3d_properties([tz])

    return agent_lines + agent_points + target_points

anim = FuncAnimation(fig, update, frames=max_steps, interval=100, blit=False)
HTML(anim.to_jshtml())
