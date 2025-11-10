# Style Guide for Language-Model-SAEs

Language-Model-SAEs basically takes advantage of Python and TypeScript (React), respectively for the core library & backend, and the frontend visualization. This style guide is a list of common _dos_ and _don'ts_.

## Python Style Guide

The Python style guide mainly follows the best practices listed in [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), but also contains instructions on writing tensor computation and distributed program.

### Lint and Format

Language-Model-SAEs uses [ruff](https://docs.astral.sh/ruff/) as the Python linter and formatter. `ruff` is a tool for detecting stylistic inconsistencies and potential bugs in Python source code. The formatter ensures consistent formatting throughout the codebase, including indentation, line width, trailing commas, and string quote style. The linter checks code quality, catching issues like unused variables and non-standard naming conventions. Make sure ruff is happy before committing, by running:

```bash
uv run ruff format # Run the Ruff formatter
uv run ruff check --fix # Run the Ruff linter
```

These commands will check the formatting and linting issues in the Python codes based on the rules defined in `pyproject.toml`. It will also fix all formatting problems and some fixable linting problems. You should manually check the remaining linting problems (if exists) and fix them.

We also have a [pre-commit](https://pre-commit.com/) hook configured in `.pre-commit-config.yaml`. Install the pre-commit hook by running

```bash
uv run pre-commit install
```

This should automatically run the above `ruff` formatter and linter checks before committing.

### Imports and Exports

- Use `import x` for importing packages and modules.
- Use `from x import y` where `x` is the package prefix and `y` is the module name with no prefix.
- Use `from x import y as z` in any of the following circumstances:
  - Two modules named `y` are to be imported.
  - `y` conflicts with a top-level name defined in the current module.
  - `y` conflicts with a common parameter name that is part of the public API (e.g., `features`).
  - `y` is an inconveniently long name.
  - `y` is too generic in the context of your code (e.g., `from storage.file_system import options as fs_options`).
- Use `import y as z` only when z is a standard abbreviation (e.g., `import numpy as np`).
- Always use complete absolute path to import first-party modules.
- For any functions or classes intended to be exposed to users, add them to `__all__` in `__init__.py`.

### Mutable States

Immutability produces code that's easier to reason about, easier to test, and easier to verify for correctness. Immutable data doesn't change, so we can safely reuse it without worrying that results will differ between calls. Immutable data can also be safely passed between threads without race conditions or other concurrency issues. We should avoid mutable states whenever possible.

#### Mutable Global States

Avoid mutable global states in the core library, as they significantly compromise the purity of core functionalities. Some global caches are permitted in the visualization server.

#### Mutable Local States

While a purely functional style (which avoids mutable states entirely) is preferred, some mutability is pragmatic or even necessary. Completely eliminating mutability may lead to overly complicated program structures and decreased readability. Thus, the preference for immutability follows a _best effort_ principle. Below are some cases where mutable states are acceptable, though immutable alternatives should be considered first:

- Use **list/dictionary/set comprehension** to create container types without resorting to procedural loops, `map`, `filter`, etc. The comprehension approach removes temporary mutable states and keeps codes concise.

  !!! success "List Comprehension"

        ```python
        arr = [x + 1 for x in range(5)]
        ```

  !!! failure "Loops"

        ```python
        arr = [] # Create an empty container
        for x in range(5):
            arr.append(x + 1) # Modify the container
        ```

  !!! failure "Explicit Map"

        ```python
        arr = list(map(lambda x: x + 1, range(5)))
        ```

- The principle of avoiding "empty first, then fill" applies to other cases as well:

  !!! success "Stack Tensors"

        ```python
        full = torch.stack([part_a, part_b])
        ```

  !!! failure "Fill Empty Tensor"

        ```python
        full = torch.zeros(2, 5, 5)
        full[0] = part_a
        full[1] = part_b
        ```

  !!! success "Concatenate Strings"

        ```python
        message = f"Error: {error_type} at line {line_num}"
        # or
        parts = ["Processing", filename, "with", str(num_items), "items"]
        message = " ".join(parts)
        ```

  !!! failure "Accumulate Strings"

        ```python
        message = "Error: "
        message += error_type
        message += " at line "
        message += str(line_num)
        ```

  !!! success "Build Dictionary"

        ```python
        config = {
            "model": model_name,
            "layers": [layer.name for layer in layers],
            "params": {k: v for k, v in params.items() if v is not None}
        }
        ```

  !!! failure "Incrementally Fill Dictionary"

        ```python
        config = {}
        config["model"] = model_name
        config["layers"] = []
        for layer in layers:
            config["layers"].append(layer.name)
        config["params"] = {}
        for k, v in params.items():
            if v is not None:
                config["params"][k] = v
        ```

- Avoid in-place modifications: create new containers instead of modifying old.

  !!! success "Create New List on Modification"

        ```python
        arr = list(range(5))
        arr = [*arr, 5]
        # or
        arr = arr + [5]
        ```

  !!! failure "Modify Old List"

        ```python
        arr = list(range(5))
        arr.append(5)
        ```

  !!! success "Create New Dictionary on Modification"

        ```python
        config = {
            "model": model_name,
            "layer": 1
        }

        config = {
            **config,
            "layer": 2
        }

        # or

        config = config | {
            "layer": 2
        }
        ```

  !!! failure "Modify Old Dictionary"

        ```python
        config = {
            "model": model_name,
            "layer": 1
        }
        config["layer"] = 2
        ```

- When mutable state is inevitable, limit its scope and preserve the purity of the outer function. Ensure that only a minimal portion of the code has access to the mutable state.

  !!! success "Localized Mutable State"

        ```python
        def compute_statistics(data: list[float]) -> dict[str, float]:
            """Pure function that returns statistics without side effects."""
            # Mutable state is confined within this function
            stats = {}
            total = 0.0

            for value in data:
                total += value

            stats["mean"] = total / len(data)
            stats["sum"] = total

            return stats  # Return new object, no external mutation
        ```

  !!! failure "Leaked Mutable State"

        ```python
        # Global mutable state
        accumulated_stats = {}

        def compute_statistics(data: list[float]) -> None:
            """Impure function that mutates global state."""
            total = 0.0
            for value in data:
                total += value

            # Mutates external state - breaks purity
            accumulated_stats["mean"] = total / len(data)
            accumulated_stats["sum"] = total
        ```

  The function remains **referentially transparent**: given the same input, it always produces the same output without observable side effects. Internal mutability for performance is acceptable as long as it doesn't leak outside the function boundary.

### Tensor Computation

### Type Annotation

All codes should be annotated with [type hints](https://docs.python.org/3/library/typing.html). Language-Model-SAEs relys on [basedpyright](https://github.com/DetachHead/basedpyright) to perform static type checking. Below lists some extra rules:

- Type hints of generic types should follow [PEP 585](https://peps.python.org/pep-0585/). Use built-in types `list`, `dict`, `set`, etc. rather than types from the `typing` module.

  !!! success "Built-in Types in Type Hints"

        ```python
        def find(haystack: dict[str, list[int]]) -> int:
            ...
        ```

  !!! failure "Types from `typing` module"

        ```python
        def find(haystack: typing.Dict[str, typing.List[int]]) -> int:
            ...
        ```

- Type hints of union types should follow [PEP 604](https://peps.python.org/pep-0604/) syntax. Use `X | Y` rather than `Union[X, Y]`, and `X | None` rather than `Optional[X]`.

  !!! success "PEP 604 Syntax"

        ```python
        def f(param: int | None) -> float | str:
            ...
        ```

  !!! failure "Old Syntax"

        ```python
        def f(param: Optional[int]) -> Union[float, str]:
            ...
        ```

- Tensors with known shapes should be annotated with `jaxtyping`.

  !!! success "Tensors with Shapes Annotated"

        ```python
        def encode(x: Float[torch.Tensor, "batch n_context d_model"]) -> Float[torch.Tensor, "batch n_context d_sae"]:
            ...
        ```

  !!! failure "Bare Tensor"

        ```python
        def encode(x: torch.Tensor) -> torch.Tensor:
            ...
        ```

Some of the type hints in the current codebase may not follow the above rules since it's heavy work to fix them all. We expect new codes to follow these rules.

## TypeScript Style Guide

TBD
