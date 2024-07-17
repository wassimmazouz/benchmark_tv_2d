from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from deepinv.optim.data_fidelity import L2


class Solver(BaseSolver):
    name = 'Chambolle-Pock'

    parameters = {
        'inner': [True, False],
        'alpha': [0.05, 0.1, 0.2],
        'gamma': [0.5, 1]
    }

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-30, patience=1000, strategy='iteration'
    )

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
        tau = self.alpha / self.gamma
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
                xk = data_fidelity.prox(xk - tau*vk, y, self.A.physics)
                tmp = vk + self.gamma * (2 * xk - x_prev)
                vk = tmp - self.gamma * prior.prox(tmp/self.gamma,
                                                   gamma=reg/self.gamma)

            else:
                prior = dinv.optim.L1Prior()
                L = dinv.optim.TVPrior().nabla
                L_adjoint = dinv.optim.TVPrior().nabla_adjoint

                x_prev = xk.clone()
                vk = L(xk)

                xk = data_fidelity.prox(xk - tau*L_adjoint(vk), y,
                                        self.A.physics)
                tmp = vk + self.gamma * L(2*xk-x_prev)
                vk = tmp - self.gamma*prior.prox(tmp/self.gamma,
                                                 gamma=reg/self.gamma)

        self.out = xk.clone()
        self.out = self.out.squeeze()

    def get_result(self):
        return dict(u=self.out.numpy())
