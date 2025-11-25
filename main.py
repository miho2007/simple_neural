#guess the nmber neural network with 1 neuron.
#-----------------------------------------------------------------------------------


import random




# simplest neural network: 1 neuron
secret = random.random()
y1 = 0
y2 = 0


i = 0
learning_rate = 0.1
# input
x = 0.7

# weight and bias
w1 = 1.2
b1 = -0.4

# second neuron weight and bias
w2 = 0.5
b2 = 0.1
# activation function (sigmoid)
import math
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

while abs(y2 - secret) > 0.0001:  # realistic tolerance
    # neuron 1
    y1 = sigmoid(w1 * x + b1)

    # neuron 2
    y2 = sigmoid(w2 * y1 + b2)

    # adjust neuron 1
    if y1 > secret:
        w1 -= learning_rate * x
        b1 -= learning_rate
    else:
        w1 += learning_rate * x
        b1 += learning_rate

    # adjust neuron 2
    if y2 > secret:
        w2 -= learning_rate * y1
        b2 -= learning_rate
    else:
        w2 += learning_rate * y1
        b2 += learning_rate

print(f"Secret number: {secret:.6f}")
print(f"Neuron 1 output: {y1:.6f}")
print(f"Neuron 2 output: {y2:.6f}")
