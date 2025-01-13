class DiscreteMapper:
    def __init__(self) -> None:
        """Initialize a new PythonDiscreteMapper with empty mappings."""
        self.value_to_int: dict[str, int] = {}
        self.int_to_value: list[str] = []
        self.counter: int = 0

    def encode(self, values: list[str]) -> list[int]:
        """Encode a list of strings to their corresponding integer indices.

        Args:
            values: List of strings to encode

        Returns:
            List of integer indices
        """
        result = []
        for value in values:
            if value not in self.value_to_int:
                self.value_to_int[value] = self.counter
                self.int_to_value.append(value)
                self.counter += 1
            result.append(self.value_to_int[value])
        return result

    def decode(self, integers: list[int]) -> list[str]:
        """Decode a list of integers back to their corresponding strings.

        Args:
            integers: List of integer indices to decode

        Returns:
            List of decoded strings

        Raises:
            IndexError: If any integer is out of range
        """
        return [self.int_to_value[i] for i in integers]

    def get_mapping(self) -> dict[str, int]:
        """Get the current mapping from strings to integers.

        Returns:
            Dictionary mapping strings to their integer indices
        """
        return self.value_to_int.copy()


class KeyedDiscreteMapper:
    def __init__(self) -> None:
        """Initialize a new PythonKeyedDiscreteMapper with empty mappers."""
        self.mappers: dict[str, DiscreteMapper] = {}

    def encode(self, key: str, values: list[str]) -> list[int]:
        """Encode a list of strings using the mapper associated with the given key.

        Args:
            key: The key identifying which mapper to use
            values: List of strings to encode

        Returns:
            List of integer indices
        """
        if key not in self.mappers:
            self.mappers[key] = DiscreteMapper()
        return self.mappers[key].encode(values)

    def decode(self, key: str, integers: list[int]) -> list[str]:
        """Decode a list of integers using the mapper associated with the given key.

        Args:
            key: The key identifying which mapper to use
            integers: List of integer indices to decode

        Returns:
            List of decoded strings

        Raises:
            KeyError: If the key doesn't exist
            IndexError: If any integer is out of range
        """
        if key not in self.mappers:
            raise KeyError("Key not found")
        return self.mappers[key].decode(integers)

    def keys(self) -> list[str]:
        """Get all keys currently in use.

        Returns:
            List of keys
        """
        return list(self.mappers.keys())
