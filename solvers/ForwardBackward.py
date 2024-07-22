from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch


class Solver(BaseSolver):
    name = 'Forward-Backward'

    parameters = {
        'gamma_mult': [0.1, 0.5, 1, 1.9]
    }

    def skip(self, A, Anorm2, reg, delta, data_fit, y, isotropy):
        if data_fit == 'huber':
            return True, f"solver does not work with {data_fit} loss"
        elif isotropy != 'anisotropic':
            return True, f"solver does not work with {isotropy} regularization"
        return False, None

    def set_objective(self, A, Anorm2, reg, delta, data_fit, y, isotropy):
        self.A, self.reg, self.y = A, reg, torch.from_numpy(y)
        self.delta, self.data_fit = delta, data_fit
        self.isotropy = isotropy
        self.Anorm2 = Anorm2

    def run(self, n_iter):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        y = self.y
        x = y.clone().to(device)
        xk = x.unsqueeze(0)

        data_fidelity = dinv.optim.L2()
        prior = dinv.optim.TVPrior()
        self.gamma = self.gamma_mult / self.Anorm2

        for _ in range(n_iter):
            xk = xk - self.gamma * data_fidelity.grad(xk, y, self.A.physics)
            xk = prior.prox(xk,  gamma=self.gamma*self.reg)

        self.out = xk.clone().to(device)
        self.out = self.out.squeeze()

    def get_result(self):
        return dict(name=f'Forward-Backward[gamma={self.gamma}]',
                    u=self.out.numpy())
