from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

# Define type variables for input and output iterable containers
InputT = TypeVar("InputT", bound=Iterable | None)
OutputT = TypeVar("OutputT", bound=Iterable)


class BaseActivationProcessor(Generic[InputT, OutputT], ABC):
    """Base class for activation processors.

    An activation processor transforms a stream of input data into a stream of output data.
    Common use cases include batching, filtering, and transforming activation data from language models.

    The processor can be used either by calling its `process()` method directly or by using the instance
    as a callable via `__call__()`.

    TypeVars:
        InputT: The type of input iterable container (e.g. List, Generator, Dataset)
        OutputT: The type of output iterable container (e.g. DataLoader, Generator)
    """

    @abstractmethod
    def process(self, data: InputT, **kwargs) -> OutputT:
        """Process the input data stream and return transformed output stream.

        Args:
            data: Input data stream to process
            **kwargs: Additional keyword arguments for processing

        Returns:
            Processed output data stream

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError

    def __call__(self, data: InputT, **kwargs) -> OutputT:
        """Process data by calling the processor instance directly.

        This is a convenience wrapper around the process() method.

        Args:
            data: Input data stream to process
            **kwargs: Additional keyword arguments for processing
        Returns:
            Processed output data stream
        """
        return self.process(data, **kwargs)
