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
            w_list = []
            d_p_list = []
            lr = group['lr']
            alpha = group['alpha']
            kappa_t = group['kappa_t']
            delta = lr / alpha / kappa_t
            gamma = (1 - alpha)/(1 + alpha)
            lr2 = (lr - alpha * delta)/(1 + alpha)

            for p in group['params']:
                params_with_grad.append(p)
                w_list.append(p)
                #v_list.append(p)
                d_p_list.append(p.grad)
                state = self.state[p]

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                w_t = w_list[i]

                w_t1 = param - lr*d_p
                param.data = (1 + gamma)*w_t1.detach() - gamma*w_t - lr2*d_p

                w_list[i] = w_t1

            for p in params_with_grad:
                state = self.state[p]

        return loss