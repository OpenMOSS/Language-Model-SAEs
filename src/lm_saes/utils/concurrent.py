from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import Iterable, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")


class BackgroundGenerator(Iterable[T]):
    """Compute elements of a generator in a background thread pool.

    This class is optimized for scenarios where either the generator or the main thread
    performs GIL-releasing work (e.g., File I/O, network requests). Best used when the
    generator has no side effects on the program state.
    """

    def __init__(
        self,
        generator: Iterable[T],
        max_prefetch: int = 1,
        executor: Optional[ThreadPoolExecutor] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the background generator.

        Args:
            generator: The generator to compute elements from in a background thread.
            max_prefetch: Maximum number of elements to precompute ahead. If <= 0,
                the queue size is infinite. When max_prefetch elements have been
                computed, the background thread will wait for consumption.
            executor: Optional ThreadPoolExecutor to use. If None, creates a single-thread executor.
        """
        self.queue = Queue(max_prefetch)
        self.generator = generator
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._owned_executor = executor is None
        self._future: Optional[Future] = None
        self.continue_iteration = True
        self._started = False
        self.pbar = tqdm(total=max_prefetch, desc=f"Background Processing {name}", smoothing=0.001, miniters=1)
        self._start()

    def _process_generator(self) -> None:
        """Process the generator items in the background thread."""
        try:
            for item in self.generator:
                if not self.continue_iteration:
                    break
                self.queue.put((True, item))
                self.pbar.update(1)
        except Exception as e:
            self.queue.put((False, e))
            self.pbar.update(1)
        finally:
            self.queue.put((False, StopIteration))
            self.pbar.update(1)

    def _start(self) -> None:
        """Start the background processing."""
        self._future = self._executor.submit(self._process_generator)

    def __next__(self) -> T:
        """Get the next item from the generator.

        Returns:
            The next item from the generator.

        Raises:
            StopIteration: When the generator is exhausted.
            Exception: Any exception raised by the generator.
        """
        if self.continue_iteration:
            success, next_item = self.queue.get()
            self.pbar.update(-1)
            if success:
                return next_item
            else:
                self.continue_iteration = False
                raise next_item
        else:
            raise StopIteration

    def __iter__(self) -> "BackgroundGenerator[T]":
        """Return self as iterator.

        Returns:
            Self as iterator.
        """
        return self

    def close(self) -> None:
        """Close the generator and clean up resources."""
        self.continue_iteration = False
        if self._future is not None:
            self._future.cancel()
        # Clear the queue to unblock any waiting threads
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.pbar.update(-1)
            except Exception:
                pass
        if self._owned_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self.pbar.close()

    def __del__(self) -> None:
        """Clean up resources when the object is deleted."""
        self.close()
