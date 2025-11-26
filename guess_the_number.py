import random
import math

# -------------------------------------------
# Activation + derivative
# -------------------------------------------
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# -------------------------------------------
# Initialize weights & biases
# -------------------------------------------
w1 = random.uniform(-1, 1)
b1 = random.uniform(-1, 1)

w2 = random.uniform(-1, 1)
b2 = random.uniform(-1, 1)

learning_rate = 0.3
x = 0.7   # constant input, as in your original code

# -------------------------------------------
# Train for many rounds
# -------------------------------------------
for round_idx in range(10):

    # new target each round
    secret = random.random()

    for step in range(5000):

        # -----------------------------
        # Forward pass
        # -----------------------------
        z1 = w1 * x + b1
        y1 = sigmoid(z1)

        z2 = w2 * y1 + b2
        y2 = sigmoid(z2)   # prediction

        # -----------------------------
        # Loss (Mean Squared Error)
        # -----------------------------
        loss = (y2 - secret) ** 2

        # -----------------------------
        # Backpropagation
        # -----------------------------

        # dLoss/dy2
        dL_dy2 = 2 * (y2 - secret)

        # dLoss/dz2
        dL_dz2 = dL_dy2 * sigmoid_derivative(z2)

        # Gradients for w2 and b2
        dL_dw2 = dL_dz2 * y1
        dL_db2 = dL_dz2

        # Now backpropagate to neuron 1
        dL_dy1 = dL_dz2 * w2

        dL_dz1 = dL_dy1 * sigmoid_derivative(z1)

        # Gradients for w1 and b1
        dL_dw1 = dL_dz1 * x
        dL_db1 = dL_dz1

        # -----------------------------
        # Update weights
        # -----------------------------
        w2 -= learning_rate * dL_dw2
        b2 -= learning_rate * dL_db2

        w1 -= learning_rate * dL_dw1
        b1 -= learning_rate * dL_db1

    print(f"Round {round_idx+1}: secret={secret:.3f} | guess={y2:.3f} | loss={loss:.6f}")
