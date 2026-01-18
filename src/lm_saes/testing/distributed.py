import functools
import importlib.util
import inspect
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _test_wrapper(rank, world_size, backend, func_name, module_path, args, kwargs):
    """
    Internal wrapper that runs on each spawned process.
    Handles distributed initialization and teardown.
    """
    # Setup environment for distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Initialize process group
    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        # Re-import the module to get the function
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise RuntimeError(f"Could not find module {module_path}")
        if spec.loader is None:
            raise RuntimeError(f"Could not find loader for module {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        func = getattr(module, func_name)
        # The re-imported func is the decorated wrapper.
        # When called here, RANK is set, so it will execute the original function.
        func(*args, **kwargs)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def distributed_test(nproc_per_node=1, backend="gloo"):
    """
    Decorator for distributed tests.
    When called from a non-distributed context (like pytest runner), it spawns nproc_per_node processes.
    When called from a distributed context (child process), it executes the test function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.environ.get("RANK") is not None:
                # We are already in a distributed process, run the test
                return func(*args, **kwargs)
            else:
                # We are in the parent process (pytest), spawn children
                module = inspect.getmodule(func)
                if module is None or module.__file__ is None:
                    raise RuntimeError("Could not determine module path for distributed test")

                module_path = os.path.abspath(module.__file__)
                func_name = func.__name__

                # mp.spawn will raise an exception if any child process fails,
                # which pytest will catch as a test failure.
                mp.spawn(
                    _test_wrapper,
                    args=(nproc_per_node, backend, func_name, module_path, args, kwargs),
                    nprocs=nproc_per_node,
                    join=True,
                )

        setattr(wrapper, "is_distributed_test", True)
        setattr(wrapper, "nproc_per_node", nproc_per_node)
        setattr(wrapper, "backend", backend)
        return wrapper

    return decorator
