import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Set

import torch


class TimerNode:
    """A node in the hierarchical timer tree.

    Attributes:
        name: The name of this timer node (without path).
        path: The full path of this node in the tree.
        total_time: Total accumulated time for this node.
        count: Number of times this timer has been called.
        parent_path: Path of the parent node in the hierarchy.
        children: Set of child node paths.
        start_time: Start time if currently running.
    """

    def __init__(self, name: str, path: str, parent_path: Optional[str] = None):
        self.name = name
        self.path = path
        self.total_time = 0.0
        self.count = 0
        self.parent_path = parent_path
        self.children: Set[str] = set()
        self.start_time: Optional[float] = None


class Timer:
    """A singleton timer class to track time usage hierarchically in different parts of the training process.

    This class provides methods to track time usage in different parts of the training process,
    organized in a true tree structure where nodes with the same name but different parents
    are recorded separately. An implicit root timer captures the entire session.

    Attributes:
        _instance: The singleton instance of the Timer class.
        _nodes: Dictionary mapping timer paths to their TimerNode objects.
        _active_stack: Stack of currently active timer paths (for hierarchy).
        _enabled: Whether the timer is enabled.
        _root_start_time: Start time of the implicit root timer.
        _session_started: Whether a timing session has been started.
    """

    _instance = None
    ROOT_NAME = "__root__"
    PATH_SEPARATOR = "/"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timer, cls).__new__(cls)
            cls._instance._nodes = {}
            cls._instance._active_stack = []
            cls._instance._enabled = False
            cls._instance._root_start_time = None
            cls._instance._session_started = False
        return cls._instance

    def _get_node_path(self, name: str) -> str:
        """Get the full path for a node based on current active stack.

        Args:
            name: The name of the timer.

        Returns:
            The full path for the node.
        """
        if name == self.ROOT_NAME:
            return self.ROOT_NAME

        if not self._active_stack:
            # Direct child of root
            return f"{self.ROOT_NAME}{self.PATH_SEPARATOR}{name}"
        else:
            # Child of current active timer
            parent_path = self._active_stack[-1]
            return f"{parent_path}{self.PATH_SEPARATOR}{name}"

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
            root_path = self.ROOT_NAME
            self._nodes[root_path] = TimerNode(self.ROOT_NAME, root_path)

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

        # Ensure root timer is started
        self._ensure_root_started()

        # Get the full path for this node
        node_path = self._get_node_path(name)

        if node_path in self._nodes and self._nodes[node_path].start_time is not None:
            raise ValueError(f"Timer {node_path} is already running")

        # Synchronize CUDA operations before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.npu.is_available():  # type: ignore
            torch.npu.synchronize()  # type: ignore

        # Determine parent path
        parent_path = None
        if self._active_stack:
            parent_path = self._active_stack[-1]
        else:
            # If no active stack, make this a child of root
            parent_path = self.ROOT_NAME

        # Create or get node
        if node_path not in self._nodes:
            self._nodes[node_path] = TimerNode(name, node_path, parent_path)
            # Add to parent's children
            if parent_path in self._nodes:
                self._nodes[parent_path].children.add(node_path)

        self._nodes[node_path].start_time = time.perf_counter()
        self._active_stack.append(node_path)

    def stop(self, name: str):
        """Stop a timer.

        Args:
            name: The name of the timer.
        """
        if not self._enabled:
            return

        # Get the expected path for the most recent timer
        if not self._active_stack:
            raise ValueError("No active timers to stop")

        expected_path = self._active_stack[-1]
        expected_name = self._nodes[expected_path].name

        if expected_name != name:
            raise ValueError(f"Timer {name} is not the most recently started timer (expected {expected_name})")

        # Synchronize CUDA operations before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.npu.is_available():  # type: ignore
            torch.npu.synchronize()  # type: ignore

        node = self._nodes[expected_path]
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

    def reset_timer(self, name: str, parent_context: Optional[str] = None):
        """Reset a specific timer and all its children.

        Args:
            name: The name of the timer.
            parent_context: Optional parent context to specify which instance of the timer to reset.
        """
        if name == self.ROOT_NAME:
            # Reset the entire session
            self.reset()
            return

        # Find the node path
        if parent_context:
            node_path = f"{parent_context}{self.PATH_SEPARATOR}{name}"
        else:
            # Find all nodes with this name
            matching_paths = [path for path in self._nodes.keys() if self._nodes[path].name == name]
            if len(matching_paths) == 1:
                node_path = matching_paths[0]
            elif len(matching_paths) > 1:
                raise ValueError(f"Multiple timers with name '{name}' found. Specify parent_context.")
            else:
                return  # No timer found

        if node_path not in self._nodes:
            return

        node = self._nodes[node_path]

        # Reset children first
        for child_path in list(node.children):
            self._reset_node_by_path(child_path)

        # Remove from parent's children
        if node.parent_path and node.parent_path in self._nodes:
            self._nodes[node.parent_path].children.discard(node_path)

        # Remove from active stack if present
        if node_path in self._active_stack:
            self._active_stack.remove(node_path)

        # Remove the node
        del self._nodes[node_path]

    def _reset_node_by_path(self, node_path: str):
        """Reset a node by its path."""
        if node_path not in self._nodes:
            return

        node = self._nodes[node_path]

        # Reset children first
        for child_path in list(node.children):
            self._reset_node_by_path(child_path)

        # Remove from parent's children
        if node.parent_path and node.parent_path in self._nodes:
            self._nodes[node.parent_path].children.discard(node_path)

        # Remove from active stack if present
        if node_path in self._active_stack:
            self._active_stack.remove(node_path)

        # Remove the node
        del self._nodes[node_path]

    def get_time(self, name: str, parent_context: Optional[str] = None) -> float:
        """Get the accumulated time for a timer.

        Args:
            name: The name of the timer.
            parent_context: Optional parent context to specify which instance of the timer.

        Returns:
            The accumulated time in seconds.
        """
        if parent_context:
            node_path = f"{parent_context}{self.PATH_SEPARATOR}{name}"
            return self._nodes[node_path].total_time if node_path in self._nodes else 0.0
        else:
            # Sum all instances with this name
            total = 0.0
            for path, node in self._nodes.items():
                if node.name == name:
                    total += node.total_time
            return total

    def get_count(self, name: str, parent_context: Optional[str] = None) -> int:
        """Get the number of times a timer has been called.

        Args:
            name: The name of the timer.
            parent_context: Optional parent context to specify which instance of the timer.

        Returns:
            The number of times the timer has been called.
        """
        if parent_context:
            node_path = f"{parent_context}{self.PATH_SEPARATOR}{name}"
            return self._nodes[node_path].count if node_path in self._nodes else 0
        else:
            # Sum all instances with this name
            total = 0
            for path, node in self._nodes.items():
                if node.name == name:
                    total += node.count
            return total

    def get_average_time(self, name: str, parent_context: Optional[str] = None) -> float:
        """Get the average time for a timer.

        Args:
            name: The name of the timer.
            parent_context: Optional parent context to specify which instance of the timer.

        Returns:
            The average time in seconds.
        """
        total_time = self.get_time(name, parent_context)
        total_count = self.get_count(name, parent_context)
        return total_time / total_count if total_count > 0 else 0.0

    def get_all_timers(self) -> Dict[str, float]:
        """Get all timers.

        Returns:
            Dictionary mapping timer paths to their accumulated time.
        """
        return {path: node.total_time for path, node in self._nodes.items()}

    def get_all_counts(self) -> Dict[str, int]:
        """Get all counts.

        Returns:
            Dictionary mapping timer paths to their call counts.
        """
        return {path: node.count for path, node in self._nodes.items()}

    def get_all_average_times(self) -> Dict[str, float]:
        """Get all average times.

        Returns:
            Dictionary mapping timer paths to their average time.
        """
        return {path: (node.total_time / node.count if node.count > 0 else 0.0) for path, node in self._nodes.items()}

    def get_active_timers(self) -> List[str]:
        """Get all currently active timers.

        Returns:
            List of active timer paths in stack order.
        """
        return self._active_stack.copy()

    def _format_node(self, node_path: str, depth: int = 0, parent_time: Optional[float] = None) -> List[str]:
        """Format a timer node and its children for display.

        Args:
            node_path: The path of the timer node.
            depth: Current depth in the hierarchy.
            parent_time: Total time of the parent node for percentage calculation.

        Returns:
            List of formatted strings for this node and its children.
        """
        if node_path not in self._nodes:
            return []

        node = self._nodes[node_path]

        # Skip displaying the root node itself, but process its children
        if node_path == self.ROOT_NAME:
            result = []
            # Sort children by total time (descending)
            sorted_children = sorted(node.children, key=lambda x: self._nodes[x].total_time, reverse=True)

            # Recursively format children with root time as parent time
            for child_path in sorted_children:
                result.extend(self._format_node(child_path, depth, node.total_time))

            return result

        indent = "  " * depth

        # Calculate percentage relative to parent
        if parent_time and parent_time > 0:
            percentage = (node.total_time / parent_time) * 100
            percentage_str = f"{percentage:.2f}% of parent"
        else:
            percentage_str = "root"

        avg_time = node.total_time / node.count if node.count > 0 else 0

        # Show just the name, not the full path for cleaner display
        result = [
            f"{indent}{node.name}: {node.total_time:.4f}s total, {avg_time:.6f}s avg "
            f"({node.count} calls), {percentage_str}"
        ]

        # Sort children by total time (descending)
        sorted_children = sorted(node.children, key=lambda x: self._nodes[x].total_time, reverse=True)

        # Recursively format children
        for child_path in sorted_children:
            result.extend(self._format_node(child_path, depth + 1, node.total_time))

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
