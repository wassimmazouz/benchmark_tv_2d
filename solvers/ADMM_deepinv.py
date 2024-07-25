from benchopt import BaseSolver, safe_import_context
import torch

with safe_import_context() as import_ctx:
    import deepinv as dinv
    from benchmark_utils.deepinv_funcs import L12Prior
    import torch.optim as optim


def func(A, L, x, y, yk, vk, gamma):
    diff = L(x) - yk + vk
    dtf = 0.5 * torch.sum(diff ** 2)
    diff2 = A.operator(x) - y
    pen = torch.sum(diff2 ** 2) / (2 * gamma)
    return dtf + pen


class Solver(BaseSolver):
    name = 'ADMM DeepInv'
    parameters = {'gamma': [0.5, 1, 2]}

    def skip(self, A, Anorm2, reg, delta, data_fit, y, isotropy):
        if data_fit == 'huber':
            return True, f"solver does not work with {data_fit} loss"
        elif isotropy == 'anisotropic':
            return True, f"solver does not work with {isotropy} regularization"
        return False, None

    def set_objective(self, A, Anorm2, reg, delta, data_fit, y, isotropy):
        self.A, self.reg, self.y = A, reg, torch.from_numpy(y)
        self.delta, self.data_fit = delta, data_fit
        self.isotropy = isotropy
        self.Anorm2 = Anorm2

    def run(self, n_iter):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y = self.y.clone().to(device)
        y = y.unsqueeze(0)
        L = dinv.optim.TVPrior().nabla
        prior = L12Prior()

        xk = torch.zeros_like(y, device=device).requires_grad_()
        yk = torch.zeros_like(L(xk), device=device)
        vk = torch.zeros_like(yk, device=device)
        optimizer = optim.LBFGS([xk])

        def closure():
            optimizer.zero_grad()
            loss = func(self.A, L, xk, y, yk, vk, self.gamma)
            loss.backward(retain_graph=True)
            return loss

        for _ in range(n_iter):
            optimizer.step(closure)

            yk = prior.prox(vk + L(xk), gamma=self.reg/self.gamma)

            vk += L(xk) - yk

        self.out = xk.detach().squeeze(0).clone().to('cpu')

    def get_result(self):
        return dict(name=f'ADMM DeepInv[gamma={self.gamma}]',
                    u=self.out.numpy())
