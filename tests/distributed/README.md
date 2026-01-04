# Distributed Tests

This directory contains tests for distributed functionality of `lm-saes`. These tests are managed by `pytest` but executed in multiple processes.

## Running Tests

Simply run `pytest` as you normally would:

```bash
pytest tests/distributed/
```

You can also run a specific test file:

```bash
pytest tests/distributed/test_load_state_dict.py
```

## How it works

The `@distributed_test` decorator in `tests/distributed/lib.py` detects if it is being run by `pytest`. If so, it uses `torch.multiprocessing.spawn` to create the number of processes specified in the decorator. 

Each child process:
1.  Initializes the distributed environment (`RANK`, `WORLD_SIZE`, etc.).
2.  Sets up the process group (`dist.init_process_group`).
3.  Executes the original test function.
4.  Cleans up the process group.

If any child process fails (raises an assertion or error), the parent process (pytest) will catch the failure and report it as a standard test failure.

## Creating New Tests

To create a new distributed test, use the `@distributed_test` decorator:

```python
from lm_saes.testing import distributed_test
import torch.distributed as dist

@distributed_test(nproc_per_node=2, backend="gloo")
def test_my_feature():
    # Distributed setup is already done
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    # Your test logic here
```
