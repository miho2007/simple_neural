import math
import matplotlib.pyplot as plt
import random
import time

# -------------------------------
# Target
# -------------------------------
target_x, target_y = 5.0, 5.0

# -------------------------------
# Neuron weights and biases
# -------------------------------
w_x, b_x = 1.0, 0.0
w_y, b_y = 1.0, 0.0
learning_rate = 0.05
scale = 0.5  # movement scaling

# -------------------------------
# Sigmoid
# -------------------------------
def sigmoid(z):
    z = max(min(z, 100), -100)
    return 1 / (1 + math.exp(-z))

# -------------------------------
# Plot setup
# -------------------------------
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Live Neural Path-Finder: Exploratory Launches")

target_dot, = ax.plot([target_x], [target_y], 'ro', markersize=8)

# -------------------------------
# Training rounds
# -------------------------------
num_rounds = 5

for round_idx in range(1, num_rounds + 1):
    # Start fresh each round
    x, y = 0.0, 0.0
    x_positions = [x]
    y_positions = [y]

    agent_dot, = ax.plot([x], [y], 'bo', markersize=8)
    trail_line, = ax.plot([x], [y], 'b-', lw=2, alpha=0.6)

    print(f"\n--- Round {round_idx} start ---")

    while abs(x - target_x) > 0.05 or abs(y - target_y) > 0.05:
        # Optional: small exploration noise
        noise_x = random.uniform(-0.05, 0.05)
        noise_y = random.uniform(-0.05, 0.05)

        # Compute neuron outputs
        dx = sigmoid(w_x * x + b_x) + noise_x
        dy = sigmoid(w_y * y + b_y) + noise_y

        # Move agent
        move_x = (dx - 0.5) * scale
        move_y = (dy - 0.5) * scale
        x += move_x
        y += move_y

        # Online learning
        if x > target_x:
            w_x -= learning_rate * x
            b_x -= learning_rate
        else:
            w_x += learning_rate * x
            b_x += learning_rate

        if y > target_y:
            w_y -= learning_rate * y
            b_y -= learning_rate
        else:
            w_y += learning_rate * y
            b_y += learning_rate

        # Update plot
        x_positions.append(x)
        y_positions.append(y)
        agent_dot.set_data([x], [y])
        trail_line.set_data(x_positions, y_positions)
        plt.draw()
        plt.pause(0.05)

        # Print live info
        print(f"x={x:.3f}, y={y:.3f} w_x={w_x:.3f} b_x={b_x:.3f} w_y={w_y:.3f} b_y={b_y:.3f}")

    # Remove agent and trail for next launch
    agent_dot.remove()
    trail_line.remove()
    plt.pause(0.2)
