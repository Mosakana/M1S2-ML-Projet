import sys
import unittest
import torch
from LinearModel import Linear
from Loss import MSELoss


class MyTestCase(unittest.TestCase):
    def test_check_gradient(self):
        batch = 64
        data_size = 100
        output_size = 50

        X = torch.randn((batch, data_size), requires_grad=True)
        w = torch.randn((data_size, output_size), requires_grad=True)
        y = torch.randn((batch, output_size), requires_grad=True)

        mse = MSELoss()
        yhat = X.mm(w)
        loss = torch.linalg.norm(y - yhat, axis=1) ** 2
        loss.sum().backward()
        delta_MSE = mse.backward(y, yhat)

        gradient_X = delta_MSE.mm(w.t())
        gradient_w = X.t().mm(delta_MSE)


        self.assertEqual(torch.all(gradient_X.isclose(X.grad, atol = 1e-4)), True)
        self.assertEqual(torch.all(gradient_w.isclose(w.grad, atol = 1e-4)), True) # add assertion here


if __name__ == '__main__':
    unittest.main()
