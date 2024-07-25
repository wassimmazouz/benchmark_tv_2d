from benchopt import BaseSolver, safe_import_context
import torch

with safe_import_context() as import_ctx:
    import deepinv as dinv
    from benchmark_utils.deepinv_funcs import L12Prior


class Solver(BaseSolver):
    name = 'ADMM DeepInv'
    parameters = {'gamma': [0.1, 0.5, 1]}

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
        y = self.y.to(device).unsqueeze(0)
        L = dinv.optim.TVPrior().nabla
        prior = L12Prior()
        xk = torch.zeros_like(y, device=device, requires_grad=True)
        yk = torch.zeros_like(L(xk), device=device)
        vk = torch.zeros_like(yk, device=device)

        optimizer = torch.optim.LBFGS(
            [xk], lr=1, max_iter=n_iter, tolerance_grad=1e-10,
            tolerance_change=1e-10)

        def closure():
            optimizer.zero_grad()
            diff = L(xk) - yk + vk
            dtf = 0.5 * torch.sum(diff ** 2)
            diff2 = self.A.operator(xk) - y
            pen = torch.sum(diff2 ** 2) / (2 * self.gamma)
            loss = dtf + pen
            return loss

        for _ in range(n_iter):
            optimizer.step(closure)
            with torch.no_grad():
                yk = prior.prox(vk + L(xk), gamma=self.reg / self.gamma)
                vk += L(xk) - yk

        self.out = xk.detach().squeeze(0).clone().to('cpu')

    def get_result(self):
        return dict(name=f'ADMM DeepInv[gamma={self.gamma}]',
                    u=self.out.numpy())
