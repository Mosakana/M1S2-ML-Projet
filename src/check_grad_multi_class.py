import unittest
import torch
import numpy as np
from Loss import CELossSoftMax

class MyTestCase(unittest.TestCase):
    def test_check_grad(self):
        data_size = 3
        N = 2
        X = torch.randn(data_size, N, requires_grad=True)

        y = torch.randint(N, ([data_size]))

        loss = (-X[torch.arange(X.shape[0]), y] + torch.log(torch.exp(X).sum(axis=1))).mean()
        loss.sum().backward()


        ce = CELossSoftMax()

        my_grad = ce.backward(y.detach().numpy(), X.detach().numpy())

        self.assertEqual(torch.all(torch.tensor(my_grad).isclose(X.grad)), True)


if __name__ == '__main__':
    unittest.main()
