#guess the nmber neural network with 1 neuron.
#-----------------------------------------------------------------------------------





# simplest neural network: 1 neuron
secret = 1.0
y = 0
i = 0
learning_rate = 0.1
# input
x = 0.7

# weight and bias
w = 1.2
b = -0.4

# activation function (sigmoid)
import math
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

while abs(y - secret) > 0.001:  # you can try more iterations
    # 1. compute neuron output
    z = w * x + b
    y = sigmoid(z)

    # 2. print current guess


    # 3. adjust weight and bias
    if y > secret:
        w -= learning_rate * x
        b -= learning_rate
    elif y < secret:
        w += learning_rate * x
        b += learning_rate

print(f"my guess is {y}")
