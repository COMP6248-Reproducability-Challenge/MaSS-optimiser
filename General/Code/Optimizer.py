import torch
from torch.optim import Optimizer



class MaSS(Optimizer):
    def __init__(self, params, lr=0, alpha=0, kappa_t=0):
        if lr and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if alpha < 0.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if kappa_t < 0.0:
            raise ValueError("Invalid kappa_t value: {}".format(kappa_t))

        defaults = dict(lr=lr, alpha=alpha, kappa_t=kappa_t)

        super(MaSS, self).__init__(params, defaults)