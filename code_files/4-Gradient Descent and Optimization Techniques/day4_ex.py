import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Visualize data
# plt.scatter(X, y, color="blue")
# plt.title('Generated Dataset')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.grid()
# plt.show()

# Initialize parameters
m = 100
theta = np.random.rand(2, 1)
learning_rate = 0.1
iterations = 1000

# Add bias term to X
X_b = np.c_[np.ones((m, 1)), X]

# Graident Descent
for iteration in range(iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

print("Optimized Parameters (Theta): \n ", theta)

import tensorflow as tf

# Prepare Data
X_tensor = tf.constant(X, dtype=tf.float32)
y_tensor = tf.constant(y, dtype=tf.float32)

# Define model
class LinearModel(tf.Module):
    def __init__(self):
        self.weights = tf.Variable(tf.random.normal([1]))
        self.bias = tf.Variable(tf.random.normal([1]))
    
    def __call__(self, X):
        return self.weights * X + self.bias
    
# Define loss function
def mse_loss(y_true, y_predict):
    return tf.reduce_mean(tf.square(y_true - y_predict))

# Train with SGD
model = LinearModel()
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# for epoch in range(100):
#     with tf.GradientTape() as tape:
#         y_pred = model(X_tensor)
#         loss = mse_loss(y_tensor, y_pred)
#     gradients = tape.gradient(loss, [model.weights, model.bias])
#     optimizer.apply_gradients(zip(gradients, [model.weights, model.bias]))
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

import torch
import torch.nn as nn
import torch.optim as optim

# Prepare data
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)

# Define model
class LinearModelTorch(nn.Module):
    def __init__(self):
        super(LinearModelTorch, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)
    
model_torch = LinearModelTorch()

# Define Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model_torch.parameters(), lr=0.1)

# train model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model_torch(X_torch)
    loss = criterion(outputs, y_torch)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")