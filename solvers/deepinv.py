from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from deepinv.optim.data_fidelity import L2


class Solver(BaseSolver):
    name = 'deepinv'

    parameters = {
        'inner': [True, False]
    }

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if data_fit == 'huber':
            return True, f"solver does not work with {data_fit} loss"
        elif isotropy == 'anisotropic':
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
        reg = self.reg
        x = y.clone().to(device)
        x = x.unsqueeze(0)
        xk = x.unsqueeze(0)
        data_fidelity = L2()
        if self.inner:
            vk = torch.zeros_like(xk)
            prior = dinv.optim.TVPrior()

        for _ in range(n_iter):
            if self.inner:
                x_prev = xk.clone()
                xk = data_fidelity.prox(xk - 0.2*vk, y, self.A.physics)
                tmp = vk + 0.5 * (2 * xk - x_prev)
                vk = tmp - 0.5 * prior.prox(2*tmp, gamma=2*reg)

            else:
                prior = dinv.optim.L1Prior()
                L = dinv.optim.TVPrior().nabla
                L_adjoint = dinv.optim.TVPrior().nabla_adjoint

                x_prev = xk.clone()
                vk = L(xk)

                xk = data_fidelity.prox(xk - 0.2*L_adjoint(vk), y,
                                        self.A.physics)
                tmp = vk + 0.5 * L(2*xk-x_prev)
                vk = tmp - 0.5*prior.prox(2*tmp, gamma=2*reg)

        self.out = xk.clone()
        self.out = self.out.squeeze()

    def get_result(self):
        return dict(u=self.out.numpy())
