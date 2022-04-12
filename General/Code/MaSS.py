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

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]

