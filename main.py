#guess the nmber neural network with 1 neuron.
#-----------------------------------------------------------------------------------





# simplest neural network: 1 neuron
secret = 1.0
y = 0
y2 = 0


i = 0
learning_rate = 0.1
# input
x = 0.7

# weight and bias
w = 1.2
b = -0.4

# second neuron weight and bias
w2 = 0.5
b2 = 0.1
# activation function (sigmoid)
import math
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

while abs(y - secret) > 0.0000000000000001:  # you can try more iterations
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

print(f"neural 1: {y}")


while abs(y2 - secret) > 0.0000000000000001:  # you can try more iterations
    # 1. compute neuron output
    z2 = w2 * y + b2
    y2 = sigmoid(z2)

    # 2. print current guess


    # 3. adjust weight and bias
    if y2 > secret:
        w2 -= learning_rate * y
        b2 -= learning_rate
    elif y2 < secret:
        w2 += learning_rate * y
        b2 += learning_rate





print(f"neural 2: {y2}")
