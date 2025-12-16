"""
Took the LR scheduler from: https://github.com/jbloomAus/DecisionTransformerInterpretability/blob/ee55df35cdb92e81d689c72fb9dd5a7252893363/src/decision_transformer/utils.py#L425
"""

import math
from typing import Any, Iterable, Optional, cast

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from lm_saes.utils.distributed.ops import item


#  None
#  Linear Warmup and decay
#  Cosine Annealing with Warmup
#  Cosine Annealing with Warmup / Restarts
def get_scheduler(scheduler_name: Optional[str], optimizer: optim.Optimizer, **kwargs: Any):
    """
    Loosely based on this, seemed simpler write this than import
    transformers: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules

    Args:
        scheduler_name (Optional[str]): Name of the scheduler to use. If None, returns a constant scheduler
        optimizer (optim.Optimizer): Optimizer to use
        **kwargs: Additional arguments to pass to the scheduler including warm_up_steps,
            training_steps, num_cycles, lr_end.
    """

    def get_smoothing_lambda(
        training_steps: int, warm_up_steps: int, gamma: float, cool_down_steps: int, lr_end: float
    ):
        smooth_steps = gamma * warm_up_steps

        def lr_lambda(steps: int):
            if steps < smooth_steps:
                return 2 * (steps + 1) / (warm_up_steps * (1 + gamma))
            elif steps < warm_up_steps:
                return 1 - ((steps / warm_up_steps - 1) ** 2) / (1 - gamma**2)
            elif steps < cool_down_steps:
                return 1.0
            else:
                progress = (steps - cool_down_steps) / (training_steps - cool_down_steps)
                return lr_end + 0.5 * (1 - lr_end) * (1 + math.cos(math.pi * progress))

        return lr_lambda

    def get_warmup_lambda(warm_up_steps: int, training_steps: int):
        def lr_lambda(steps: int):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                return (training_steps - steps) / (training_steps - warm_up_steps)

        return lr_lambda

    # heavily derived from hugging face although copilot helped.
    def get_warmup_cosine_lambda(warm_up_steps: int, training_steps: int, lr_end: float):
        def lr_lambda(steps: int):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                progress = (steps - warm_up_steps) / (training_steps - warm_up_steps)
                return lr_end + 0.5 * (1 - lr_end) * (1 + math.cos(math.pi * progress))

        return lr_lambda

    def get_warmup_exp_lambda(warm_up_steps: int, training_steps: int, lr_end: float):
        def lr_lambda(steps: int):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                return math.pow(lr_end, (steps - warm_up_steps) / (training_steps - warm_up_steps))

        return lr_lambda

    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name.lower() == "constantwithwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        cool_down_steps = kwargs.get("cool_down_steps", 0)
        training_steps = kwargs.get("training_steps")
        lr_end_ratio = kwargs.get("lr_end_ratio", 0.0)

        assert training_steps is not None, "training_steps must be provided"
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(
                1.0,
                (steps + 1) / warm_up_steps,
                lr_end_ratio + (1 - lr_end_ratio) / cool_down_steps * max(training_steps - steps, 1),  # type: ignore
            ),
        )
    elif scheduler_name.lower() == "constantwithwarmupsmooth":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        assert training_steps is not None, "training_steps must be provided"
        cool_down_steps = training_steps - int(1.5 * warm_up_steps)
        assert training_steps is not None, "training_steps must be provided"
        lr_lambda = get_smoothing_lambda(training_steps, warm_up_steps, 0.5, cool_down_steps, 0.0)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "linearwarmupdecay":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        assert training_steps is not None, "training_steps must be provided"
        lr_lambda = get_warmup_lambda(warm_up_steps, training_steps)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealing":
        training_steps = kwargs.get("training_steps")
        assert training_steps is not None, "training_steps must be provided"
        eta_min = kwargs.get("lr_end", 0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps, eta_min=eta_min)
    elif scheduler_name.lower() == "cosineannealingwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        assert training_steps is not None, "training_steps must be provided"
        eta_min = kwargs.get("lr_end", 0)
        lr_lambda = get_warmup_cosine_lambda(warm_up_steps, training_steps, eta_min)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealingwarmrestarts":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        num_cycles = kwargs.get("num_cycles", 1)
        T_0 = training_steps // num_cycles
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)
    elif scheduler_name.lower() == "exponentialwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        assert training_steps is not None, "training_steps must be provided"
        eta_min = kwargs.get("lr_end", 1 / 32)
        lr_lambda = get_warmup_exp_lambda(warm_up_steps, training_steps, eta_min)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


class SparseAdam(Optimizer):
    """
    Implements SparseAdam algorithm, which only updates parameters and their momentum
    when their gradients are non-zero.

    This optimizer is specifically designed for Sparse Autoencoders (SAE) training,
    with the following key features:
    1. Only updates parameters and momentum when gradients are non-zero
    2. Prevents gradient amplification for long-inactive features when they suddenly activate

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if isinstance(grad, DTensor):
                    # TODO: Figure out how this bug is triggered
                    # Workaround for a DTensor bug with unclear trigger conditions.
                    # Without this, when computing exp_avg_sq, the result of (1 - beta2) * grad * grad
                    # appears to be incorrectly cached as (1 - beta1) * grad * grad (i.e., reusing
                    # the intermediate result from exp_avg computation).
                    grad = grad.redistribute(p.device_mesh, p.placements)
                if not grad.any():
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Momentum buffer
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Squared momentum buffer
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Only update momentum for non-zero gradients
                mask = grad != 0

                # Update momentum
                exp_avg = exp_avg * beta1
                exp_avg = torch.where(mask, exp_avg + (1 - beta1) * grad, exp_avg)

                # Update squared momentum
                exp_avg_sq = exp_avg_sq * beta2
                exp_avg_sq = torch.where(mask, exp_avg_sq + (1 - beta2) * grad * grad, exp_avg_sq)

                # Save back to state
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                # Compute step size
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Apply updates
                denom = exp_avg_sq.sqrt().add_(eps)
                update = torch.where(mask, step_size * exp_avg / denom, torch.zeros_like(p))
                p.data.sub_(update)

        return loss


def compute_grad_norm(
    params: Iterable[torch.nn.Parameter],
    device_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    gradients = [
        p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad for p in list(params) if p.grad is not None
    ]
    assert len(gradients) > 0, "No gradients found"

    local_total_norm = torch.stack([g.norm(2) for g in gradients]).norm(2)
    local_nonzero_count = cast(torch.Tensor, sum((g != 0).sum() for g in gradients))

    if device_mesh is not None:
        local_sq_norm = local_total_norm.square()
        dist.all_reduce(local_sq_norm, op=dist.ReduceOp.SUM, group=device_mesh.get_group("model"))
        dist.all_reduce(local_nonzero_count, op=dist.ReduceOp.SUM, group=device_mesh.get_group("model"))
        total_norm = local_sq_norm.sqrt()
    else:
        total_norm = local_total_norm

    return total_norm / local_nonzero_count.sqrt()


def clip_grad_norm(
    params: Iterable[torch.nn.Parameter],
    max_norm: float,
    device_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    params = list(params)
    grad_norm = compute_grad_norm(params, device_mesh)
    if item(grad_norm) > max_norm:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(max_norm / (item(grad_norm) + 1e-6))

    return grad_norm
