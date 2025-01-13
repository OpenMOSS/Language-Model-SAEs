from lm_saes.utils.discrete import DiscreteMapper, KeyedDiscreteMapper


def test_discrete_mapper():
    mapper = DiscreteMapper()
    assert mapper.encode(["a", "b", "a", "c"]) == [0, 1, 0, 2]
    assert mapper.decode([0, 1, 0, 2]) == ["a", "b", "a", "c"]

    assert mapper.get_mapping() == {"a": 0, "b": 1, "c": 2}

    assert mapper.encode(["a", "c", "d"]) == [0, 2, 3]
    assert mapper.decode([0, 2, 3]) == ["a", "c", "d"]
    assert mapper.get_mapping() == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert mapper.encode(["a", "c", "d", "a"]) == [0, 2, 3, 0]
    assert mapper.decode([0, 2, 3, 0]) == ["a", "c", "d", "a"]
    assert mapper.get_mapping() == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_keyed_discrete_mapper():
    mapper = KeyedDiscreteMapper()
    assert mapper.encode("foo", ["a", "b", "a", "c"]) == [0, 1, 0, 2]
    assert mapper.decode("foo", [0, 1, 0, 2]) == ["a", "b", "a", "c"]
    assert mapper.keys() == ["foo"]

    assert mapper.encode("bar", ["a", "c", "d"]) == [0, 1, 2]
    assert mapper.decode("bar", [0, 1, 2]) == ["a", "c", "d"]
    assert sorted(mapper.keys()) == ["bar", "foo"]


if __name__ == "__main__":
    import timeit

    print(timeit.timeit("test_discrete_mapper()", globals=globals(), number=1000))
    print(timeit.timeit("test_keyed_discrete_mapper()", globals=globals(), number=1000))
