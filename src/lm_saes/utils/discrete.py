from typing import Any, Optional, cast

import torch


class DiscreteMapper:
    def __init__(self) -> None:
        """Initialize a new PythonDiscreteMapper with empty mappings."""
        self.value_to_int: dict[str, int] = {}
        self.int_to_value: list[str] = []
        self.counter: int = 0

    def _dist_encode(self, values: list[str], group: torch.distributed.ProcessGroup) -> list[int]:
        local_rank = torch.distributed.get_rank(group)
        not_seen_values = [value for value in values if value not in self.value_to_int]
        not_seen_values_list = [None] * group.size()
        torch.distributed.all_gather_object(not_seen_values_list, not_seen_values, group=group)
        not_seen_values_list = [
            sublist for sublist in not_seen_values_list if (sublist is not None and len(sublist) > 0)
        ]
        flattened_not_seen_values = [item for sublist in not_seen_values_list for item in cast(list[str], sublist)]

        # get unique values
        unique_values = list(set(flattened_not_seen_values))
        if len(unique_values) == 0:
            return [self.value_to_int[value] for value in values]
        update_hash = {}
        counter = self.counter
        broadcast_list: list[Any] = [None] * (len(unique_values) + 1) if local_rank != 0 else []
        if local_rank == 0:
            for value in unique_values:
                assert value not in self.value_to_int
                update_hash[value] = counter
                broadcast_list.append(value)
                counter += 1
            broadcast_list.append(update_hash)
        torch.distributed.broadcast_object_list(broadcast_list, group=group, group_src=0)
        self.value_to_int |= broadcast_list[-1]
        self.int_to_value.extend(broadcast_list[:-1])
        self.counter += len(broadcast_list[:-1])

        # check if all hash_table is the same
        check_list = [None] * group.size()
        torch.distributed.all_gather_object(check_list, self.value_to_int, group=group)
        assert all(check_list[i] == check_list[0] for i in range(1, group.size())), (
            "value_to_int is not consistent across processes"
        )  # TODO: Remove this check for speed up
        return [self.value_to_int[value] for value in values]

    def encode(self, values: list[str], group: Optional[torch.distributed.ProcessGroup] = None) -> list[int]:
        """Encode a list of strings to their corresponding integer indices.

        Args:
            values: List of strings to encode

        Returns:
            List of integer indices
        """
        if group is not None:
            return self._dist_encode(values, group)
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

    def encode(self, key: str, values: list[str], group: Optional[torch.distributed.ProcessGroup] = None) -> list[int]:
        """Encode a list of strings using the mapper associated with the given key.

        Args:
            key: The key identifying which mapper to use
            values: List of strings to encode

        Returns:
            List of integer indices
        """
        if key not in self.mappers:
            self.mappers[key] = DiscreteMapper()
        return self.mappers[key].encode(values, group=group)

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
