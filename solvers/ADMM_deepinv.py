from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch


class Solver(BaseSolver):
    name = 'ADMM DeepInv'

    parameters = {
        'tau': [0.1, 1, 10],
        'rho': [0.1, 1, 10]
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

        y = self.y.to(device)
        x = torch.zeros_like(y, device=device)
        z = torch.zeros_like(y, device=device)
        u = torch.zeros_like(y, device=device)

        data_fidelity = dinv.optim.L2()
        L = dinv.optim.TVPrior().nabla
        L_adjoint = dinv.optim.TVPrior().nabla_adjoint
        prior = dinv.optim.TVPrior()

        for _ in range(n_iter):
            # Update x
            x = data_fidelity.prox(z - u, y, self.A.physics, gamma=self.tau)

            # Update z
            z = L(x + u)
            z = prior.prox(z, gamma=self.reg / self.tau)
            z = L_adjoint(z)

            # Update dual variable u
            u += x - z

        self.out = x.clone().to(device)
        self.out = self.out.squeeze()

    def get_result(self):
        return dict(name=f'ADMM[tau={self.tau},rho={self.rho}]',
                    u=self.out.numpy())
