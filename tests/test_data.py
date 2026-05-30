"""Pure-logic tests for table serialization, TabFact parsing, and sampling."""

from src.data import (
    disjoint_train_eval_indices,
    parse_corpus_table,
    sample_indices,
    serialize_table,
)


def test_serialize_table_header_then_rows():
    table = {"header": ["Name", "Score"], "rows": [["Alice", "92"], ["Bob", "78"]]}
    out = serialize_table(table)
    assert out == "Name | Score\nAlice | 92\nBob | 78"


def test_serialize_table_casts_non_strings():
    table = {"header": ["a", 1], "rows": [[1, 2.5], [True, None]]}
    out = serialize_table(table)
    # first line is the header, every cell stringified
    assert out.splitlines()[0] == "a | 1"
    assert "2.5" in out


def test_serialize_empty_rows():
    table = {"header": ["x", "y"], "rows": []}
    assert serialize_table(table) == "x | y"


def test_parse_corpus_table_from_list():
    table = [["name", "wins"], ["Lions", "10"], ["Tigers", "7"]]
    parsed = parse_corpus_table(table)
    assert parsed["header"] == ["name", "wins"]
    assert parsed["rows"] == [["Lions", "10"], ["Tigers", "7"]]


def test_parse_corpus_table_from_string_repr():
    parsed = parse_corpus_table("[['a', 'b'], ['1', '2']]")
    assert parsed["header"] == ["a", "b"]
    assert parsed["rows"] == [["1", "2"]]


def test_parse_corpus_table_empty():
    assert parse_corpus_table([]) == {"header": [], "rows": []}


def test_disjoint_train_eval_no_overlap():
    train, ev = disjoint_train_eval_indices(
        total=200, eval_n=20, eval_seed=13, train_n=50, train_seed=42
    )
    assert len(ev) == 20
    assert len(train) == 50
    assert set(train).isdisjoint(set(ev))  # the guarantee
    assert all(0 <= i < 200 for i in train + ev)


def test_disjoint_eval_slice_is_seed_stable():
    _, ev1 = disjoint_train_eval_indices(200, 20, 13, 50, 1)
    _, ev2 = disjoint_train_eval_indices(200, 20, 13, 50, 999)
    assert ev1 == ev2  # eval slice depends only on eval_seed, not train_seed


def test_disjoint_train_caps_at_pool_size():
    train, ev = disjoint_train_eval_indices(30, 10, 13, 999, 42)
    assert len(train) == 20  # remainder after the 10 eval indices
    assert set(train).isdisjoint(set(ev))


def test_sample_indices_is_deterministic_and_sorted():
    a = sample_indices(100, 10, seed=13)
    b = sample_indices(100, 10, seed=13)
    assert a == b
    assert a == sorted(a)
    assert len(a) == 10
    assert all(0 <= i < 100 for i in a)


def test_sample_indices_take_all_when_n_exceeds_total():
    assert sample_indices(5, 10, seed=1) == [0, 1, 2, 3, 4]


def test_sample_indices_different_seeds_differ():
    assert sample_indices(1000, 20, seed=1) != sample_indices(1000, 20, seed=2)
