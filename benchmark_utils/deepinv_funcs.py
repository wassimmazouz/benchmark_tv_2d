import deepinv as dinv
import torch


class DeepInverseOperator:
    def __init__(self, tensor_size, mask=0.5, sigma=0.2):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        self.operator = dinv.physics.Inpainting(
            tensor_size=tensor_size,
            mask=mask,
            device=device
        )
        self.physics = self.operator
        self.physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

    def __matmul__(self, u):
        x = torch.from_numpy(u).unsqueeze(0)
        x = x.unsqueeze(0)
        return self.operator(x).squeeze().numpy()

    def apply_physics(self, u):
        return self.physics(u)

    def adjoint(self, u):
        return self.operator.A_adjoint(u)

    @property
    def T(self):
        return DeepInverseOperatorAdjoint(self)


class DeepInverseOperatorAdjoint:
    def __init__(self, deep_inv_op):
        self.deep_inv_op = deep_inv_op

    def __matmul__(self, u):
        return self.deep_inv_op.adjoint(u)
