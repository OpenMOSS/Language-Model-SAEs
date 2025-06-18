import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Set

import torch


class TimerNode:
    """A node in the hierarchical timer tree.

    Attributes:
        name: The name of this timer node.
        total_time: Total accumulated time for this node.
        count: Number of times this timer has been called.
        parent: Parent node in the hierarchy.
        children: Set of child node names.
        start_time: Start time if currently running.
    """

    def __init__(self, name: str, parent: Optional["TimerNode"] = None):
        self.name = name
        self.total_time = 0.0
        self.count = 0
        self.parent = parent
        self.children: Set[str] = set()
        self.start_time: Optional[float] = None

        if parent:
            parent.children.add(name)


class Timer:
    """A singleton timer class to track time usage hierarchically in different parts of the training process.

    This class provides methods to track time usage in different parts of the training process,
    organized in a hierarchical structure where nested timers show percentages relative to their
    parent timers rather than the total time. An implicit root timer captures the entire session.

    Attributes:
        _instance: The singleton instance of the Timer class.
        _nodes: Dictionary mapping timer names to their TimerNode objects.
        _active_stack: Stack of currently active timer names (for hierarchy).
        _enabled: Whether the timer is enabled.
        _root_start_time: Start time of the implicit root timer.
        _session_started: Whether a timing session has been started.
    """

    _instance = None
    ROOT_NAME = "__root__"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timer, cls).__new__(cls)
            cls._instance._nodes = {}
            cls._instance._active_stack = []
            cls._instance._enabled = False
            cls._instance._root_start_time = None
            cls._instance._session_started = False
        return cls._instance

    def _ensure_root_started(self):
        """Ensure the root timer is started for the session."""
        if not self._session_started and self._enabled:
            # Synchronize CUDA operations before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.npu.is_available():  # type: ignore
                torch.npu.synchronize()  # type: ignore

            self._root_start_time = time.perf_counter()
            self._session_started = True

            # Create root node
            self._nodes[self.ROOT_NAME] = TimerNode(self.ROOT_NAME)

    def _finalize_root(self):
        """Finalize the root timer if there are no active timers."""
        if (
            self._session_started
            and not self._active_stack
            and self._root_start_time is not None
            and self.ROOT_NAME in self._nodes
        ):
            # Synchronize CUDA operations before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.npu.is_available():  # type: ignore
                torch.npu.synchronize()  # type: ignore

            root_node = self._nodes[self.ROOT_NAME]
            if root_node.start_time is None:  # Only finalize if not already finalized
                elapsed = time.perf_counter() - self._root_start_time
                root_node.total_time = elapsed
                root_node.count = 1

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

        if name == self.ROOT_NAME:
            raise ValueError(f"Timer name '{self.ROOT_NAME}' is reserved for the root timer")

        if name in self._nodes and self._nodes[name].start_time is not None:
            raise ValueError(f"Timer {name} is already running")

        # Ensure root timer is started
        self._ensure_root_started()

        # Synchronize CUDA operations before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.npu.is_available():  # type: ignore
            torch.npu.synchronize()  # type: ignore

        # Determine parent
        parent_node = None
        if self._active_stack:
            parent_name = self._active_stack[-1]
            parent_node = self._nodes[parent_name]
        else:
            # If no active stack, make this a child of root
            parent_node = self._nodes[self.ROOT_NAME]

        # Create or get node
        if name not in self._nodes:
            self._nodes[name] = TimerNode(name, parent_node)
        else:
            assert self._nodes[name].parent == parent_node, f"Timer {name} has a different parent"

        self._nodes[name].start_time = time.perf_counter()
        self._active_stack.append(name)

    def stop(self, name: str):
        """Stop a timer.

        Args:
            name: The name of the timer.
        """
        if not self._enabled:
            return

        if name not in self._nodes or self._nodes[name].start_time is None:
            raise ValueError(f"Timer {name} is not running")

        if not self._active_stack or self._active_stack[-1] != name:
            raise ValueError(f"Timer {name} is not the most recently started timer")

        # Synchronize CUDA operations before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.npu.is_available():  # type: ignore
            torch.npu.synchronize()  # type: ignore

        node = self._nodes[name]
        elapsed = time.perf_counter() - node.start_time
        node.total_time += elapsed
        node.count += 1
        node.start_time = None
        self._active_stack.pop()

        # Finalize root timer if this was the last active timer
        self._finalize_root()

    def reset(self):
        """Reset all timers."""
        self._nodes = {}
        self._active_stack = []
        self._root_start_time = None
        self._session_started = False

    def reset_timer(self, name: str):
        """Reset a specific timer and all its children.

        Args:
            name: The name of the timer.
        """
        if name == self.ROOT_NAME:
            # Reset the entire session
            self.reset()
            return

        if name not in self._nodes:
            return

        node = self._nodes[name]

        # Reset children first
        for child_name in list(node.children):
            self.reset_timer(child_name)

        # Remove from parent's children
        if node.parent:
            node.parent.children.discard(name)

        # Remove from active stack if present
        if name in self._active_stack:
            self._active_stack.remove(name)

        # Remove the node
        del self._nodes[name]

    def get_time(self, name: str) -> float:
        """Get the accumulated time for a timer.

        Args:
            name: The name of the timer.

        Returns:
            The accumulated time in seconds.
        """
        return self._nodes[name].total_time if name in self._nodes else 0.0

    def get_count(self, name: str) -> int:
        """Get the number of times a timer has been called.

        Args:
            name: The name of the timer.

        Returns:
            The number of times the timer has been called.
        """
        return self._nodes[name].count if name in self._nodes else 0

    def get_average_time(self, name: str) -> float:
        """Get the average time for a timer.

        Args:
            name: The name of the timer.

        Returns:
            The average time in seconds.
        """
        if name not in self._nodes:
            return 0.0
        node = self._nodes[name]
        return node.total_time / node.count if node.count > 0 else 0.0

    def get_all_timers(self) -> Dict[str, float]:
        """Get all timers.

        Returns:
            Dictionary mapping timer names to their accumulated time.
        """
        return {name: node.total_time for name, node in self._nodes.items()}

    def get_all_counts(self) -> Dict[str, int]:
        """Get all counts.

        Returns:
            Dictionary mapping timer names to their call counts.
        """
        return {name: node.count for name, node in self._nodes.items()}

    def get_all_average_times(self) -> Dict[str, float]:
        """Get all average times.

        Returns:
            Dictionary mapping timer names to their average time.
        """
        return {name: self.get_average_time(name) for name in self._nodes}

    def get_active_timers(self) -> List[str]:
        """Get all currently active timers.

        Returns:
            List of active timer names in stack order.
        """
        return self._active_stack.copy()

    def _format_node(self, name: str, depth: int = 0, parent_time: Optional[float] = None) -> List[str]:
        """Format a timer node and its children for display.

        Args:
            name: The name of the timer node.
            depth: Current depth in the hierarchy.
            parent_time: Total time of the parent node for percentage calculation.

        Returns:
            List of formatted strings for this node and its children.
        """
        if name not in self._nodes:
            return []

        node = self._nodes[name]

        # Skip displaying the root node itself, but process its children
        if name == self.ROOT_NAME:
            result = []
            # Sort children by total time (descending)
            sorted_children = sorted(node.children, key=lambda x: self._nodes[x].total_time, reverse=True)

            # Recursively format children with root time as parent time
            for child_name in sorted_children:
                result.extend(self._format_node(child_name, depth, node.total_time))

            return result

        indent = "  " * depth

        # Calculate percentage relative to parent
        if parent_time and parent_time > 0:
            percentage = (node.total_time / parent_time) * 100
            percentage_str = f"{percentage:.2f}% of parent"
        else:
            percentage_str = "root"

        avg_time = node.total_time / node.count if node.count > 0 else 0

        result = [
            f"{indent}{name}: {node.total_time:.4f}s total, {avg_time:.6f}s avg ({node.count} calls), {percentage_str}"
        ]

        # Sort children by total time (descending)
        sorted_children = sorted(node.children, key=lambda x: self._nodes[x].total_time, reverse=True)

        # Recursively format children
        for child_name in sorted_children:
            result.extend(self._format_node(child_name, depth + 1, node.total_time))

        return result

    def summary(self) -> str:
        """Get a hierarchical summary of all timers.

        Returns:
            A string summarizing all timers in hierarchical format with percentages relative to parent nodes.
        """
        if not self._nodes:
            return "No timers recorded."

        # Ensure root is finalized
        self._finalize_root()

        if self.ROOT_NAME not in self._nodes:
            return "No root timer found."

        root_node = self._nodes[self.ROOT_NAME]
        result = [f"Total session time: {root_node.total_time:.4f}s"]
        result.extend(self._format_node(self.ROOT_NAME))

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
