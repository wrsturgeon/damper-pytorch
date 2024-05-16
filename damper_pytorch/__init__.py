import torch
from torch import optim
from typing import Callable, Optional, Self


hyperparameters = {
    "sensitivity": {"type": float, "default": 0.05},
    "std_update": {"type": float, "default": 0.2},
}


class Damper(optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        sensitivity: float = hyperparameters["sensitivity"]["default"],
        std_update: float = hyperparameters["std_update"]["default"],
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

                # TODO: variance instead of dividing by std twice
                a = p.grad / (eps + std)
                b = state["last_grad"] / (eps + std)
                dot_prod = a * b
                lr = state["lr"] * torch.exp(dot_prod)
                lr = torch.clamp(lr, min=eps)

                actual_variances = torch.square(p.grad)
                variances = torch.square(std)
                dLds = (variances - actual_variances) / (
                    eps + torch.abs(torch.no_grad(std))
                )
                std = std - std_update * dLds

                state["lr"] = lr
                state["std"] = std
                state["last_grad"] = p.grad

                p.add_(p, alpha=-lr)

        return loss
