import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List

import torch


class Timer:
    """A singleton timer class to track time usage in different parts of the training process.

    This class provides methods to track time usage in different parts of the training process,
    such as communication vs computation. It is designed as a singleton to be accessible
    from anywhere in the codebase.

    Attributes:
        _instance: The singleton instance of the Timer class.
        _timers: Dictionary mapping timer names to their accumulated time.
        _start_times: Dictionary mapping timer names to their start times.
        _counts: Dictionary mapping timer names to the number of times they've been called.
        _active_timers: List of currently active timers.
        _enabled: Whether the timer is enabled.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timer, cls).__new__(cls)
            cls._instance._timers = defaultdict(float)
            cls._instance._start_times = {}
            cls._instance._counts = defaultdict(int)
            cls._instance._active_timers = []
            cls._instance._enabled = False
        return cls._instance

    @contextmanager
    def time(self, name: str):
        """Context manager to time a block of code.

        Args:
            name: The name of the timer.
        """
        if not self._enabled:
            yield
            return

        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def start(self, name: str):
        """Start a timer.

        Args:
            name: The name of the timer.
        """
        if not self._enabled:
            return

        if name in self._start_times:
            raise ValueError(f"Timer {name} is already running")

        # Synchronize CUDA operations before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.npu.is_available():
            torch.npu.synchronize()

        self._start_times[name] = time.perf_counter()
        self._active_timers.append(name)

    def stop(self, name: str):
        """Stop a timer.

        Args:
            name: The name of the timer.
        """
        if not self._enabled:
            return

        if name not in self._start_times:
            raise ValueError(f"Timer {name} is not running")

        # Synchronize CUDA operations before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.npu.is_available():
            torch.npu.synchronize()

        elapsed = time.perf_counter() - self._start_times[name]
        self._timers[name] += elapsed
        self._counts[name] += 1
        del self._start_times[name]
        self._active_timers.remove(name)

    def reset(self):
        """Reset all timers."""
        self._timers = defaultdict(float)
        self._start_times = {}
        self._counts = defaultdict(int)
        self._active_timers = []

    def reset_timer(self, name: str):
        """Reset a specific timer.

        Args:
            name: The name of the timer.
        """
        if name in self._timers:
            self._timers[name] = 0.0
        if name in self._counts:
            self._counts[name] = 0
        if name in self._start_times:
            del self._start_times[name]
        if name in self._active_timers:
            self._active_timers.remove(name)

    def get_time(self, name: str) -> float:
        """Get the accumulated time for a timer.

        Args:
            name: The name of the timer.

        Returns:
            The accumulated time in seconds.
        """
        return self._timers.get(name, 0.0)

    def get_count(self, name: str) -> int:
        """Get the number of times a timer has been called.

        Args:
            name: The name of the timer.

        Returns:
            The number of times the timer has been called.
        """
        return self._counts.get(name, 0)

    def get_average_time(self, name: str) -> float:
        """Get the average time for a timer.

        Args:
            name: The name of the timer.

        Returns:
            The average time in seconds.
        """
        count = self._counts.get(name, 0)
        if count == 0:
            return 0.0
        return self._timers.get(name, 0.0) / count

    def get_all_timers(self) -> Dict[str, float]:
        """Get all timers.

        Returns:
            Dictionary mapping timer names to their accumulated time.
        """
        return dict(self._timers)

    def get_all_counts(self) -> Dict[str, int]:
        """Get all counts.

        Returns:
            Dictionary mapping timer names to their call counts.
        """
        return dict(self._counts)

    def get_all_average_times(self) -> Dict[str, float]:
        """Get all average times.

        Returns:
            Dictionary mapping timer names to their average time.
        """
        return {name: self.get_average_time(name) for name in self._timers}

    def get_active_timers(self) -> List[str]:
        """Get all currently active timers.

        Returns:
            List of active timer names.
        """
        return self._active_timers.copy()

    def summary(self) -> str:
        """Get a summary of all timers.

        Returns:
            A string summarizing all timers.
        """
        result = []
        total_time = sum(self._timers.values())

        for name, time_value in sorted(self._timers.items(), key=lambda x: x[1], reverse=True):
            count = self._counts[name]
            avg_time = time_value / count if count > 0 else 0
            percentage = (time_value / total_time * 100) if total_time > 0 else 0

            result.append(
                f"{name}: {time_value:.4f}s total, {avg_time:.6f}s avg ({count} calls), {percentage:.2f}% of total"
            )

        return "\n".join(result)

    def enable(self):
        """Enable the timer."""
        self._enabled = True

    def disable(self):
        """Disable the timer."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Whether the timer is enabled."""
        return self._enabled


# Global singleton instance
timer = Timer()
