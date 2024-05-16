import torch
from torch import optim
from typing import Optional, Self


class Damper(optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        sensitivity: float = 0.05,
        std_update: float = 0.2,
        epsilon: float = 1e-5,
    ) -> Self:

        if lr <= 0:
            raise ValueError(f"Non-positive damper learning rate {lr}")
        if sensitivity <= 0:
            raise ValueError(f"Non-positive damper sensitivity {sensitivity}")
        if std_update <= 0:
            raise ValueError(f"Non-positive damper std_update {std_update}")
        if epsilon <= 0:
            raise ValueError(f"Non-positive damper epsilon {epsilon}")

        defaults = dict(
            lr=lr,
            sensitivity=sensitivity,
            std_update=std_update,
            epsilon=epsilon,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Damper doesn't support sparse gradients")

                lr = group["lr"]
                sensitivity = group["sensitivity"]
                std_update = group["std_update"]
                eps = group["epsilon"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["lr"] = group["lr"]
                    state["std"] = torch.ones_like(p)
                    state["last_grad"] = torch.zeros_like(p)

                std = state["std"]
                actual_variances = torch.square(p.grad)
                variances = torch.square(std)
                dLds = (variances - actual_variances) / (
                    eps + torch.abs(torch.no_grad(std))
                )
                std = std - std_update * dLds

                # TODO: variance instead of dividing by std twice
                a = p.grad / (eps + std)
                b = state["last_grad"] / (eps + std)
                dot_prod = a * b
                lr = state["lr"] * torch.exp(dot_prod)
                lr = torch.maximum(lr, eps)

                state["lr"] = lr
                state["std"] = std
                state["last_grad"] = p.grad

                p.add_(p, alpha=-lr)

        return loss
