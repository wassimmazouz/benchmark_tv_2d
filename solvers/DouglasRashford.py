from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch


class Solver(BaseSolver):
    name = 'Douglas-Rashford'

    parameters = {
        'tau_mult': [0.1, 0.5, 0.9],
        'gamma': [0.1, 1, 10]
    }

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if data_fit == 'huber':
            return True, f"solver does not work with {data_fit} loss"
        elif isotropy != 'anisotropic':
            return True, f"solver does not work with {isotropy} regularization"
        return False, None

    def set_objective(self, A, reg, delta, data_fit, y, isotropy):
        self.A, self.reg, self.y = A, reg, torch.from_numpy(y)
        self.delta, self.data_fit = delta, data_fit
        self.isotropy = isotropy

    def run(self, n_iter):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        y = self.y
        x = y.clone().to(device)
        x = x.unsqueeze(0)
        xk = x.unsqueeze(0)

        data_fidelity = dinv.optim.L2()
        prior = dinv.optim.TVPrior()
        tau = self.tau_mult / self.gamma
        vk = xk

        for _ in range(n_iter):

            x_prev = xk.clone()

            xk = data_fidelity.prox(xk - tau*vk, y, self.A.physics, gamma=tau)
            tmp = vk+self.gamma*(2*xk-x_prev)
            vk = tmp - self.gamma*prior.prox(tmp/self.gamma,
                                             gamma=self.reg/self.gamma)

        self.out = xk.clone().to(device)
        self.out = self.out.squeeze()

    def get_result(self):
        return dict(u=self.out.numpy())
