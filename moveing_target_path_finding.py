import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# -----------------------------
# Activation functions
# -----------------------------
def tanh(z):
    return math.tanh(z)

def tanh_derivative(z):
    return 1 - math.tanh(z)**2

# -----------------------------
# Initialize network
# -----------------------------
input_size = 4
hidden_size = 6
output_size = 2

w1 = np.random.uniform(-1, 1, (hidden_size, input_size))
b1 = np.random.uniform(-1, 1, hidden_size)

w2 = np.random.uniform(-1, 1, (output_size, hidden_size))
b2 = np.random.uniform(-1, 1, output_size)

learning_rate = 0.05

# -----------------------------
# Forward pass
# -----------------------------
def forward(input_vec):
    z1 = w1 @ input_vec + b1
    h = np.tanh(z1)
    z2 = w2 @ h + b2
    output = z2  # dx, dy
    return z1, h, z2, output

# -----------------------------
# Training step
# -----------------------------
def train_step(x, y, tx, ty, iterations=3):
    for _ in range(iterations):
        input_vec = np.array([x, y, tx, ty])
        z1, h, z2, output = forward(input_vec)
        dx, dy = output
        x_new, y_new = x + dx, y + dy

        # Loss
        loss = (x_new - tx)**2 + (y_new - ty)**2

        # Backpropagation
        dL_dd = np.array([2*(x_new - tx), 2*(y_new - ty)])
        dL_dw2 = np.outer(dL_dd, h)
        dL_db2 = dL_dd.copy()
        dL_dh = w2.T @ dL_dd
        dL_dz1 = dL_dh * (1 - h**2)
        dL_dw1 = np.outer(dL_dz1, input_vec)
        dL_db1 = dL_dz1.copy()

        # Update weights
        w2[:] -= learning_rate * dL_dw2
        b2[:] -= learning_rate * dL_db2
        w1[:] -= learning_rate * dL_dw1
        b1[:] -= learning_rate * dL_db1

# -----------------------------
# Simulation parameters
# -----------------------------
x, y = 0.0, 0.0
positions = [(x, y)]
max_steps = 100
max_step_size = 0.2

# -----------------------------
# Moving target parameters
# -----------------------------
radius = 4.0
center = (2.5, 2.5)
theta = 0.0
dtheta = 2 * math.pi / max_steps

target_positions = []

# -----------------------------
# Simulate chasing moving target
# -----------------------------
for step in range(max_steps):
    # Move target in a circle
    tx = center[0] + radius * math.cos(theta)
    ty = center[1] + radius * math.sin(theta)
    target_positions.append((tx, ty))
    theta += dtheta

    # Train and move agent
    train_step(x, y, tx, ty, iterations=3)
    _, _, _, output = forward(np.array([x, y, tx, ty]))
    dx, dy = output

    # Limit step size
    step_size = np.linalg.norm([dx, dy])
    if step_size > max_step_size:
        dx *= max_step_size / step_size
        dy *= max_step_size / step_size

    x += dx
    y += dy
    positions.append((x, y))

# -----------------------------
# Plot animation
# -----------------------------
fig, ax = plt.subplots()
ax.set_xlim(-1, 8)
ax.set_ylim(-1, 8)
ax.grid(True)

point, = ax.plot([], [], 'bo', label='Agent')
target_point, = ax.plot([], [], 'ro', label='Target')
path_line, = ax.plot([], [], 'g-', linewidth=2, label='Path')

def update(frame):
    xs = np.array([p[0] for p in positions[:frame+1]])
    ys = np.array([p[1] for p in positions[:frame+1]])
    point.set_data([xs[-1]], [ys[-1]])
    path_line.set_data(xs, ys)
    target_point.set_data([target_positions[frame][0]], [target_positions[frame][1]])
    return point, path_line, target_point

anim = FuncAnimation(fig, update, frames=max_steps, interval=100, blit=False)

HTML(anim.to_jshtml())
