import deepinv as dinv
import torch
from deepinv.optim.prior import Prior


class DeepInverseOperator:
    def __init__(self, tensor_size, type_A='inpainting', mask=0.5, sigma=0.2):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        if type_A == 'inpainting':
            self.operator = dinv.physics.Inpainting(
                tensor_size=tensor_size,
                mask=mask,
                device=device
            )

        if type_A == 'denoising':
            self.operator = dinv.physics.Denoising(device=device)

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


class L12Prior(Prior):
    r"""
    :math:`\ell_{1,2}` prior :math:`\reg{x} = \sum_i\| x_i \|_2`.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, *args, **kwargs):
        r"""
        Computes the regularizer :math:`\reg{x} = \sum_i\| x_i \|_2`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.Tensor) prior :math:`\reg{x}`.
        """
        return torch.norm(torch.norm(x, p=2, dim=-1), p=1)

    def prox(self, x, *args, ths=1.0, gamma=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the l12 regularization term :math:`\regname` at :math:`x`.

        More precisely, it computes

        .. math::
            \operatorname{prox}_{\gamma g}(x) = (1 - \frac{\gamma}{max{\Vert x \Vert_2,\gamma}}) x


        where :math:`\gamma` is a stepsize.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return torch.Tensor: proximity operator at :math:`x`.
        """
        tmp = torch.norm(x, p=2, dim=-1)
        tmp = 1 - gamma / torch.maximum(tmp, gamma*torch.ones_like(tmp))
        tmp = torch.movedim(torch.tile(tmp, (2, 1, 1, 1, 1)), 0, -1)
        return x * tmp
