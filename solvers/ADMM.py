from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg
    from scipy.optimize import minimize
    from benchmark_utils.shared import huber
    from benchmark_utils.shared import grad_huber
    from benchmark_utils.matrix_op import div, grad
    from benchmark_utils.matrix_op import dual_prox_tv_iso
    from benchmark_utils.matrix_op import dual_prox_tv_aniso


def loss(y, A, u, delta, n, m, zh, zv, muh, muv, gamma):
    u_tmp = u.reshape((n, m))
    R = A @ u_tmp - y
    gh, gv = grad(u_tmp)
    return huber(R, delta) + gamma / 2 * (
        np.linalg.norm(gh - zh + muh / gamma, ord='fro') ** 2
        + np.linalg.norm(gv - zv + muv / gamma, ord='fro') ** 2)


def jac_loss(y, A, u, delta, n, m, zh, zv, muh, muv, gamma):
    u_tmp = u.reshape((n, m))
    R = A @ u_tmp - y
    gh, gv = grad(u_tmp)
    return (A.T @ grad_huber(R, delta)
            - gamma * div(gh - zh + muh / gamma,
                          gv - zv + muv / gamma)).flatten()


class Solver(BaseSolver):
    """Alternating direction method."""
    name = 'ADMM'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'gamma': [0.1]}

    def skip(self, A, Anorm2, reg, delta, data_fit, y, isotropy):
        if isotropy not in ["anisotropic", "isotropic"]:
            return True, "Only aniso and isoTV are implemented yet"
        return False, None

    def set_objective(self, A, Anorm2, reg, delta, data_fit, y, isotropy):
        self.reg, self.delta = reg, delta
        self.isotropy = isotropy
        self.data_fit = data_fit
        self.A, self.y = A, y
        self.Anorm2 = Anorm2

    def run(self, callback):
        c, n, m = self.y.shape
        # Initialization
        self.u = np.zeros((c, n, m))
        u = np.zeros((c, n, m))
        zh = np.zeros((c, n, m))  # We consider non-cyclic finite difference
        zv = np.zeros((c, n, m))
        muh = np.zeros((c, n, m))  # We consider non-cyclic finite difference
        muv = np.zeros((c, n, m))

        # Prox of sigma * G*, where G* is conjugate of G
        # G is reg * l1-norm
        proj = {
            'anisotropic': dual_prox_tv_aniso,
            'isotropic': dual_prox_tv_iso,
        }.get(self.isotropy, dual_prox_tv_aniso)

        gamma = self.gamma
        tol_cg = 1e-12
        print('self.y.shape:', self.y.shape)
        Aty = self.A.T @ self.y  # Flatten y to match A's input shape

        def matvec_rgb(x):
            x = x.reshape((c, n, m))
            result = np.zeros_like(x)
            for i in range(c):
                Ax = self.A @ x[i].reshape(-1)
                AtAx = self.A.T @ Ax
                gh, gv = grad(x[i])
                result[i] = AtAx.reshape((n, m)) - gamma * div(gh, gv)
            return result.ravel()

        AtA_gDtD = LinearOperator(shape=(c*n*m, c*n*m), matvec=matvec_rgb)

        while callback():
            if self.data_fit == 'lsq':
                u_tmp = (Aty + div(muh, muv) - gamma * div(zh, zv)).flatten()
                u, _ = cg(AtA_gDtD, u_tmp, x0=u.flatten(), tol=tol_cg)
                u = u.reshape((c, n, m))
            elif self.data_fit == 'huber':
                def func(u):
                    return loss(self.y, self.A, u, self.delta,
                                n, m, zh, zv, muh, muv, gamma)

                def jac(u):
                    return jac_loss(self.y, self.A, u, self.delta,
                                    n, m, zh, zv, muh, muv, gamma)

                u = minimize(func, x0=u.flatten(), jac=jac,
                             method='BFGS', tol=tol_cg).x
                u = u.reshape((c, n, m))

            for i in range(c):
                gh, gv = grad(u[i])
                zh[i], zv[i] = proj(gh * gamma + muh[i],
                                    gv * gamma + muv[i],
                                    self.reg)
                zh[i] = (gh * gamma + muh[i] - zh[i]) / gamma
                zv[i] = (gv * gamma + muv[i] - zv[i]) / gamma
                muh[i] += gamma * (gh - zh[i])
                muv[i] += gamma(gv - zv[i])

            self.u = u

    def get_result(self):
        return dict(u=self.u)
