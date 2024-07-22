from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from benchmark_utils.deepinv_funcs import L12Prior


class Solver(BaseSolver):
    name = 'Chambolle-Pock'

    parameters = {
        'tau_mult': [0.1, 0.5, 0.9],
        'gamma': [0.35, 1, 10]
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

    def run(self, n_iter):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        y = self.y
        x = y.clone().to(device)
        xk = x.unsqueeze(0)

        data_fidelity = dinv.optim.L2()
        L = dinv.optim.TVPrior().nabla
        L_adjoint = dinv.optim.TVPrior().nabla_adjoint
        prior = L12Prior()
        Lnorm2 = 8
        self.tau = self.tau_mult / (Lnorm2 * self.gamma)
        vk = L(xk)

        for _ in range(n_iter):
            x_prev = xk.clone()
            xk = data_fidelity.prox(xk - self.tau*L_adjoint(vk),
                                    y, self.A.physics, gamma=self.tau)
            tmp = vk+self.gamma*L(2*xk-x_prev)
            vk = tmp - self.gamma * prior.prox(tmp/self.gamma,
                                               gamma=self.reg/self.gamma)

        self.out = xk.clone().to(device)
        self.out = self.out.squeeze()

    def get_result(self):
        return dict(name=f'Chambolle-Pock[tau={self.tau},gamma={self.gamma}]', u=self.out.numpy())
