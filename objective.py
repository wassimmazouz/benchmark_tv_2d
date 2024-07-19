from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.shared import huber
    from benchmark_utils.matrix_op import grad


class Objective(BaseObjective):
    min_benchopt_version = "1.5"
    name = "TV2D"

    parameters = {
        'reg': [0.1, 0.5, 1, 2],
        'delta': [0.9],
        'isotropy': ["anisotropic", "isotropic"],
        'data_fit': ["lsq", "huber"]
    }

    def set_data(self, A, y, Anorm2):
        self.A = A
        self.y = y
        self.Anorm2 = Anorm2

    def evaluate_result(self, u):
        R = self.y - self.A @ u

        if self.data_fit == "lsq":
            loss = .5 * np.linalg.norm(R) ** 2
        else:
            loss = huber(R, self.delta)

        if self.isotropy == "isotropic":
            penalty = self.isotropic_tv_value(u)
        else:
            penalty = self.anisotropic_tv_value(u)

        return loss + self.reg * penalty

    def get_one_result(self):
        return dict(u=np.zeros(self.y.shape))

    def get_objective(self):
        return dict(A=self.A,
                    Anorm2=self.Anorm2,
                    reg=self.reg,
                    delta=self.delta,
                    data_fit=self.data_fit,
                    y=self.y,
                    isotropy=self.isotropy)

    def isotropic_tv_value(self, u):
        gh, gv = grad(u)
        return (np.sqrt(gh ** 2 + gv ** 2)).sum()

    def anisotropic_tv_value(self, u):
        gh, gv = grad(u)
        return (np.abs(gh) + np.abs(gv)).sum()
