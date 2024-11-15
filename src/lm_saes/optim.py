"""
Took the LR scheduler from: https://github.com/jbloomAus/DecisionTransformerInterpretability/blob/ee55df35cdb92e81d689c72fb9dd5a7252893363/src/decision_transformer/utils.py#L425
"""

import math
from typing import Any, Optional

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


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
