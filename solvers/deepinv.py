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
        'alpha': [0.05, 0.1],
        'gamma': [0.5, 0.9, 1],
        'a': [3]
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

        Anorm2 = 1
        Lnorm2 = 8
        tau = 0.9 / (Anorm2 / 2 + Lnorm2 * self.gamma)
        y = self.y
        reg = self.reg
        x = y.clone().to(device)
        x = x.unsqueeze(0)
        xk = x.unsqueeze(0)
        wk = xk.clone().to(device)
        uk = xk.clone().to(device)
        data_fidelity = L2()
        if self.inner:
            vk = torch.zeros_like(xk)
            prior = dinv.optim.TVPrior()
            a = self.a
        else:
            prior = dinv.optim.L1Prior()
            L = dinv.optim.TVPrior().nabla
            L_adjoint = dinv.optim.TVPrior().nabla_adjoint
            vk = L(xk)

        for k in range(n_iter):
            if self.inner:
                tk = (k+a-1)/a

                xk_prev = xk.clone()

                xk = wk - self.gamma*data_fidelity.grad(wk, y, self.A.physics)
                xk = prior.prox(xk, gamma=self.gamma*reg)

                wk = (1-1/tk)*xk+1/tk*uk

                uk = xk_prev+tk*(xk-xk_prev)

            else:

                x_prev = xk.clone().to(device)

                xk = xk - tau*data_fidelity.grad(xk, y,
                                                 self.A.physics) - tau*L_adjoint(vk)
                tmp = vk+self.gamma*L(2*xk-x_prev)
                vk = tmp - self.gamma*prior.prox(tmp/self.gamma, gamma=reg/self.gamma)

        self.out = xk.clone().to(device)
        self.out = self.out.squeeze()

    def get_result(self):
        return dict(u=self.out.numpy())
