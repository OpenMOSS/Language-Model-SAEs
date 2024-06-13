from typing import Any, get_origin, get_args
from dataclasses import fields as dataclass_fields, is_dataclass
import inspect

from typing_extensions import TypedDict, is_typeddict, Self

class Field(TypedDict):
    name: str
    type: Any

def fields(cls) -> list[Field]:
    assert is_dataclass(cls) or is_typeddict(cls), f"{cls} is not a dataclass or TypedDict"
    if is_dataclass(cls):
        return [Field(name=f.name, type=f.type) for f in dataclass_fields(cls)]
    else:
        return [Field(name=k, type=v) for k, v in inspect.get_annotations(cls).items()]
    
def flattened_fields(cls) -> list[Field]:
    if is_dataclass(cls) or is_typeddict(cls):
        f = []
        for field in fields(cls):
            f.extend(flattened_fields(field["type"]))
            f.append(field)
        return f
    elif get_origin(cls) == list:
        return flattened_fields(get_args(cls)[0])
    elif get_origin(cls) == dict:
        return flattened_fields(get_args(cls)[1])
    return []

def is_flattenable(cls) -> bool:
    return len(flattened_fields(cls)) > 0

def from_flattened(cls, data: Any, context: dict | None = None, path: str = "obj"):
    """Construct an object, especially a dataclass or TypedDict, from a flat structure.
    This is a superset of a traditional deserialization aimed at conveniently constructing nested dataclasses.

    The difference between this function and a traditional deserialization is that this function will further
    pass the fields in the outer dataclass to the inner dataclasses. This is useful when we want to construct
    nested dataclasses, where different subclasses hold fields with the same name and we want theses fields to
    be exactly the same.

    Args:
        cls: The class to construct.
        data: The data to construct the object.
        context: The context to pass to the inner dataclasses. Not necessary to be manually specified.
        path: The path of the current field, mainly for debugging. Not necessary to be manually specified.

    Returns:
        The constructed object.

    Examples:
    
        Construct a dataclass from a flat structure:
            >>> from dataclasses import dataclass
            >>>
            >>> @dataclass
            ... class A:
            ...     a1: int
            ...     a2: str
            ...
            >>> @dataclass
            ... class B:
            ...     a: A
            ...     b: int
            ...
            >>> from_flattened(B, {"a1": 1, "a2": "2", "b": 3}) # Construct the object B with the fields in A and B. Fields in A will automatically be passed to construct A.
            B(a=A(a1=1, a2='2'), b=3)

        Construct a dataclass with a list of dataclasses:
            >>> from dataclasses import dataclass
            >>>
            >>> @dataclass
            ... class A:
            ...     a1: int
            ...     a2: str
            ...
            >>> @dataclass
            ... class B:
            ...     a: list[A]
            ...     b: int
            ...
            >>> from_flattened(B, {"a": [{"a1": 1}, {"a1": 2}], "a2": "3", "b": 4}) # Construct the object B with the fields in A and B. Fields in A will automatically be passed to all elements in the list.
            B(a=[A(a1=1, a2='3'), A(a1=2, a2='3')], b=4)

        Construct a deep nested dataclass with default values:
            >>> from dataclasses import dataclass
            >>>
            >>> @dataclass
            ... class A:
            ...     a1: int
            ...     a2: str
            ...
            >>> @dataclass
            ... class B:
            ...     a: A
            ...     b1: int = 3
            ...
            >>> @dataclass
            ... class C:
            ...     b: B
            ...     c: int
            ...
            >>> from_flattened(C, {"a1": 1, "a2": "2", "c": 4}) # Deep nested dataclasses are also supported.
            C(b=B(a=A(a1=1, a2='2'), b1=3), c=4)
    """

    if context is None:
        context = {}
    if not is_flattenable(cls):
        # Skip further checking for non-flattenable classes
        return data
    if is_dataclass(cls) or is_typeddict(cls):
        if data == "__missing__": # Accept not specified fields and construct the object with the context.
            data = {}
        if is_dataclass(cls) and isinstance(data, cls):
            return data
        assert isinstance(data, dict), f"Field {path} is not a dict"
        data = {**context, **data}
        context = {**context, **data}
        for field in fields(cls):        
            # We have to further transform the specified data (if exists) into the specified type
            specified_data = data[field["name"]] if field["name"] in data else "__missing__" # We use __missing__ to indicate that the field is not specified. Not specified fields may have different behaviors due to their types and default values.
            f = from_flattened(field["type"], specified_data, context, f"{path}.{field['name']}")
            if f != "__missing__": # Don't update the field if it is still regarded as not specified, so that the default value can be used.
                data[field["name"]] = f
        # Remove the fields that are not in the data. This fields may exist in the child classes
        # and we don't want to pass them to the constructor of the current class.
        data = {k: v for k, v in data.items() if k in [f["name"] for f in fields(cls)]}
        return cls(**data)
    elif get_origin(cls) == list:
        if data == "__missing__":
            return "__missing__"
        assert isinstance(data, list), f"Field {path} is not a list"
        return [from_flattened(get_args(cls)[0], d, context, f"{path}.{i}") for i, d in enumerate(data)]
    elif get_origin(cls) == dict:
        if data == "__missing__":
            return "__missing__"
        assert isinstance(data, dict), f"Field {path} is not a dict"
        return {k: from_flattened(get_args(cls)[1], v, context, f"{path}.{k}") for k, v in data.items()}
    raise ValueError(f"Unexpected flattenable type {cls}. It's an internal error. Please report this issue to the developers.")

class FlattenableModel:
    @classmethod
    def from_flattened(cls, data: Any) -> Self:
        """Construct from a flat structure. This is a superset of a traditional deserialization aimed at conveniently constructing nested dataclasses.

        The difference between this function and a traditional deserialization is that this function will further
        pass the fields in the outer dataclass to the inner dataclasses. This is useful when we want to construct
        nested dataclasses, where different subclasses hold fields with the same name and we want theses fields to
        be exactly the same.

        Args:
            data: The data to construct the object.

        Returns:
            The constructed object.

        Examples:
        
            Construct a dataclass from a flat structure:
                >>> from dataclasses import dataclass
                >>>
                >>> @dataclass
                ... class A(FlattenableModel):
                ...     a1: int
                ...     a2: str
                ...
                >>> @dataclass
                ... class B(FlattenableModel):
                ...     a: A
                ...     b: int
                ...
                >>> B.from_flattened({"a1": 1, "a2": "2", "b": 3}) # Construct the object B with the fields in A and B. Fields in A will automatically be passed to construct A.
                B(a=A(a1=1, a2='2'), b=3)

            Construct a dataclass with a list of dataclasses:
                >>> from dataclasses import dataclass
                >>>
                >>> @dataclass
                ... class A(FlattenableModel):
                ...     a1: int
                ...     a2: str
                ...
                >>> @dataclass
                ... class B(FlattenableModel):
                ...     a: list[A]
                ...     b: int
                ...
                >>> B.from_flattened({"a": [{"a1": 1}, {"a1": 2}], "a2": "3", "b": 4}) # Construct the object B with the fields in A and B. Fields in A will automatically be passed to all elements in the list.
                B(a=[A(a1=1, a2='3'), A(a1=2, a2='3')], b=4)

            Construct a deep nested dataclass with default values:
                >>> from dataclasses import dataclass
                >>>
                >>> @dataclass
                ... class A(FlattenableModel):
                ...     a1: int
                ...     a2: str
                ...
                >>> @dataclass
                ... class B(FlattenableModel):
                ...     a: A
                ...     b1: int = 3
                ...
                >>> @dataclass
                ... class C(FlattenableModel):
                ...     b: B
                ...     c: int
                ...
                >>> C.from_flattened({"a1": 1, "a2": "2", "c": 4}) # Deep nested dataclasses are also supported.
                C(b=B(a=A(a1=1, a2='2'), b1=3), c=4)
        """
        return from_flattened(cls, data)