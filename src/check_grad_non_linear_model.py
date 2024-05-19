import sys
import unittest
import torch
from NonLinearModel import Tanh, Sigmoid
from Loss import MSELoss



class MyTestCase(unittest.TestCase):
    def test_check_gradient(self):
        batch = 64
        data_size = 100

        X_tanh = torch.randn((batch, data_size), requires_grad=True)
        y = torch.randn((batch, data_size))
        t = torch.nn.Tanh()

        loss = torch.linalg.norm(y - t(X_tanh), axis=1) ** 2
        loss.sum().backward()

        tanh = Tanh()
        mse = MSELoss()
        delta_X = mse.backward(y, tanh.forward(X_tanh.detach().numpy()))
        grad_tanh_X = tanh.backward_delta(X_tanh.detach().numpy(), delta_X.numpy())

        self.assertEqual(torch.all(torch.tensor(grad_tanh_X).isclose(X_tanh.grad, atol=1e-4)), True)

        X_sigmoid = torch.randn((batch, data_size), requires_grad=True)
        s = torch.nn.Sigmoid()
        loss2 = torch.linalg.norm(y - s(X_sigmoid), axis=1) ** 2
        loss2.sum().backward()

        sigmoid = Sigmoid()
        delta_X2 = mse.backward(y, sigmoid.forward(X_sigmoid.detach().numpy()))
        grad_sigmoid_X = sigmoid.backward_delta(X_sigmoid.detach().numpy(), delta_X2.numpy())

        self.assertEqual(torch.all(torch.tensor(grad_sigmoid_X).isclose(X_sigmoid.grad, atol=1e-4)), True)



