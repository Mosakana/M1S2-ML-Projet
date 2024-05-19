class Sequentiel():
    def __init__(self, *args):
        self.net = args
        self.outputs = None

    def forward(self, X):
        res = [X]
        for model in self.net:
            res.append(model.forward(res[-1]))

        self.outputs = res

        return res

    def backward(self, delta, eps):
        grads_in_nets = [delta]

        # back-propagation
        reversed_outputs = self.outputs[::-1]
        for i, model in enumerate(self.net[::-1]):
            grads_in_nets.append(model.backward_delta(reversed_outputs[i], grads_in_nets[-1]))

        # update gradients in linear models
        for i, model in enumerate(self.net):
            if model.__class__.__name__ == 'Linear':
                model.backward_update_gradient(self.outputs[i], grads_in_nets[-(i+2)])
                model.update_parameters(eps)

    def zero_grad(self):
        for model in self.net:
            model.zero_grad()

class Optim():
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps
        self.y_hat = None
        self.score = 0

    def step(self, batch_X, batch_y):
        self.net.zero_grad()
        y_hat = self.net.forward(batch_X)[-1]
        self.score += self.loss.forward(batch_y, y_hat).sum()

        self.net.backward(self.loss.backward(batch_y, y_hat), self.eps)

    def update_output(self):
        self.y_hat = self.net.outputs[-1]

    def zero_score(self):
        self.score = 0

def SGD(net, X, y, batch_size, epochs, loss, eps=1e-3):
    optim = Optim(net, loss, eps)
    n_batch = X.shape[0] // batch_size

    for iteration in range(epochs):
        optim.zero_score()
        for i in range(n_batch):
            optim.step(X[i * batch_size: (i+1) * batch_size], y[i * batch_size: (i+1) * batch_size])

        optim.step(X[n_batch * batch_size:], y[n_batch * batch_size:])
        optim.update_output()

    return optim.y_hat, optim.score