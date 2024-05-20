import unittest
from MultiClass import CELossSoftMax
import torch
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_check_grad(self):
        data_size = 5
        N = 10
        X = torch.randn(data_size, N, requires_grad=True)

        y = torch.randint(N, ([data_size]))

        loss = -X[torch.arange(X.shape[0]), y] + torch.log(torch.exp(X).sum(axis=1))
        loss.sum().backward()

        ce = CELossSoftMax()

        my_grad = ce.backward(y.detach().numpy(), X.detach().numpy())

        print(my_grad)
        print(X.grad)

        self.assertEqual(torch.all(torch.tensor(my_grad).isclose(X.grad)), True)




if __name__ == '__main__':
    unittest.main()
