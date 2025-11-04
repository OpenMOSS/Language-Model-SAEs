"""Centralized logging configuration for LM SAEs project."""

import logging
import sys
from typing import Optional

import torch.distributed as dist


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    enable_distributed: bool = True,
) -> logging.Logger:
    """Set up project-wide logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        enable_distributed: Whether to include distributed training considerations

    Returns:
        Configured logger instance
    """
    if format_string is None:
        if enable_distributed and dist.is_initialized():
            rank = dist.get_rank()
            format_string = f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Get project logger
    logger = logging.getLogger("lm_saes")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Name of the module/component

    Returns:
        Logger instance
    """
    return logging.getLogger(f"lm_saes.{name}")


def is_master() -> bool:
    """Check if this is the master process in distributed training."""
    return not dist.is_initialized() or dist.get_rank() == 0


def format_metrics_dict(metrics: dict, title: str = "Metrics", max_width: int = 80) -> str:
    """Format a metrics dictionary in a readable table-like format.

    Args:
        metrics: Dictionary of metrics to format
        title: Title for the metrics section
        max_width: Maximum width for the formatted output

    Returns:
        Formatted string representation of the metrics
    """
    if not metrics:
        return f"{title}: (empty)"

    # Group metrics by category (prefix before /)
    categories = {}
    for key, value in metrics.items():
        if "/" in key:
            category, metric = key.split("/", 1)
            if category not in categories:
                categories[category] = {}
            categories[category][metric] = value
        else:
            if "general" not in categories:
                categories["general"] = {}
            categories["general"][key] = value

    lines = [f"\n{'=' * max_width}", f"{title.center(max_width)}", "=" * max_width]

    for category, category_metrics in categories.items():
        lines.append(f"\n📊 {category.upper().replace('_', ' ')}")
        lines.append("-" * (len(category) + 5))

        # Find max key length for alignment
        max_key_len = max(len(key) for key in category_metrics.keys())

        for key, value in category_metrics.items():
            # Format the value based on its type
            if isinstance(value, float):
                if abs(value) < 0.001 and value != 0:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{value:.4f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)

            lines.append(f"  {key:<{max_key_len}} : {formatted_value}")

    lines.append("=" * max_width)
    return "\n".join(lines)


def log_metrics(logger: logging.Logger, metrics: dict, step: Optional[int] = None, title: str = "Training Metrics"):
    """Log metrics in a formatted, readable way.

    Args:
        logger: Logger instance to use
        metrics: Dictionary of metrics to log
        step: Optional step number to include
        title: Title for the metrics section
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if step is not None:
        title = f"{title} - Step {step}"

    formatted_metrics = format_metrics_dict(metrics, title)
    logger.info(formatted_metrics)


class DistributedLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that only logs from the master process in distributed training."""

    def __init__(self, logger: logging.Logger, extra: Optional[dict] = None):
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        return f"[Master] {msg}", kwargs

    def log(self, level, msg, *args, **kwargs):
        if is_master():
            super().log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if is_master():
            super().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if is_master():
            super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if is_master():
            super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if is_master():
            super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if is_master():
            super().critical(msg, *args, **kwargs)


def get_distributed_logger(name: str) -> DistributedLoggerAdapter:
    """Get a distributed-aware logger that only logs from the master process.

    Args:
        name: Name of the module/component

    Returns:
        Distributed logger adapter
    """
    logger = get_logger(name)
    return DistributedLoggerAdapter(logger)
