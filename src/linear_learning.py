from LinearModel import MSELoss, Linear
import numpy as np
import matplotlib.pyplot as plt

batch = 64
input = 100
output = 50
epochs = 100

mse_scores = []

X = np.random.randn(batch, input)
y = np.random.randn(batch, output)

linear_model = Linear(input, output)
mse = MSELoss()

yhat = linear_model.forward(X)

mse_scores.append(mse.forward(y, yhat))
delta_mse = mse.backward(y, yhat)

for i in range(epochs):
    linear_model.backward_update_gradient(X, delta_mse)
    linear_model.update_parameters()
    linear_model.zero_grad()

    yhat = linear_model.forward(X)
    mse_scores.append(mse.forward(y, yhat))

    delta_mse = mse.backward(y, yhat)

plt.figure(figsize=(10, 10))
plt.plot(np.arange(epochs+1), mse_scores)
plt.show()

