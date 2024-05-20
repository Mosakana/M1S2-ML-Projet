import sys
import unittest
import torch
from src.NonLinearModel import Tanh, Sigmoid, LogSoftMax
from src.Loss import MSELoss, BCELoss
import numpy as np



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

class TestSoftMax(unittest.TestCase):
    def test_softmax(self):
        batch = 64
        X = torch.randn(batch, requires_grad=True)
        y = torch.randn(batch, requires_grad=True)
        torch_softmax = torch.log(torch.nn.functional.softmax(X))
        loss = torch.linalg.norm(y - torch_softmax, axis=-1) ** 2
        loss.sum().backward()

        softmax = LogSoftMax()
        mse = MSELoss()
        my_softmax = softmax.forward(X.detach().numpy())
        my_grad_softmax = softmax.backward_delta(X.detach().numpy(), mse.backward(y.detach().numpy(), my_softmax))

        self.assertEqual(torch.all(torch.tensor(my_softmax).isclose(torch_softmax, atol=1e-4)), True)

        self.assertEqual(torch.all(torch.tensor(my_grad_softmax).isclose(X.grad, atol=1e-4)), True)

class TestBCE(unittest.TestCase):
    def test_bce(self):
        batch = 10
        data_size = 1

        y = torch.randint(0, 2, [batch, 1]).float()

        X_sigmoid = torch.randn((batch, data_size), requires_grad=True)
        s = torch.nn.Sigmoid()
        loss2 = torch.nn.functional.binary_cross_entropy(s(X_sigmoid), y)
        loss2.sum().backward()

        bce = BCELoss()

        sigmoid = Sigmoid()
        delta_X2 = bce.backward(y.detach().numpy(), sigmoid.forward(X_sigmoid.detach().numpy()))
        grad_sigmoid_X = sigmoid.backward_delta(X_sigmoid.detach().numpy(), delta_X2)


        self.assertEqual(torch.all(torch.tensor(bce.forward(y.detach().numpy(), sigmoid.forward(X_sigmoid.detach().numpy()))).isclose(loss2, atol=1e-4)), True)

        self.assertEqual(torch.all(torch.tensor(grad_sigmoid_X).isclose(X_sigmoid.grad, atol=1e-4)), True)









