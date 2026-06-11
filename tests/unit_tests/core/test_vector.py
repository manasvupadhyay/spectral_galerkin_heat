"""Tests for Vec3 (core/vector.py)."""

from dataclasses import FrozenInstanceError

import pytest

from fast_heat_solv.core.vector import Vec3


def test_getitem():
    v = Vec3(1, 2, 3)
    assert (v[0], v[1], v[2]) == (1, 2, 3)


def test_iter_pairs_with_zip():
    # zip(n, d) over two Vec3 is the idiom the state constructors use.
    n, d = Vec3(8, 4, 2), Vec3(0.5, 0.5, 0.5)
    assert list(zip(n, d)) == [(8, 0.5), (4, 0.5), (2, 0.5)]


def test_zyx_reverses_axis_order():
    # Solver arrays are indexed [z, y, x]; zyx() is where that reversal lives.
    # Use (1, 2, 3) so a partial swap can't slip through.
    v = Vec3(1, 2, 3)
    r = v.zyx()

    assert (r.x, r.y, r.z) == (3, 2, 1)
    assert (v.x, v.y, v.z) == (1, 2, 3)  # original untouched (Vec3 is frozen)


def test_truediv_vec():
    # size / n -> grid spacing.
    assert Vec3(4.0, 2.0, 1.0) / Vec3(8, 4, 2) == Vec3(0.5, 0.5, 0.5)


def test_truediv_scalar():
    assert Vec3(4.0, 2.0, 1.0) / 2 == Vec3(2.0, 1.0, 0.5)


def test_mul_vec():
    assert Vec3(1, 2, 3) * Vec3(4, 5, 6) == Vec3(4, 10, 18)


def test_mul_scalar():
    assert Vec3(1, 2, 3) * 2 == Vec3(2, 4, 6)


def test_map():
    assert Vec3(1, 2, 3).map(lambda c: c * c) == Vec3(1, 4, 9)


# A Vec3 built on `geom` (size/n/d) is handed out and shared by reference across
# the solver, never copied. We don't want to drops `frozen`



def test_frozen():
    # Immutable -> a shared geom.d can't be mutated through one holder and
    # corrupt every other holder. Writing a component must raise, not alias.
    v = Vec3(1, 2, 3)
    with pytest.raises(FrozenInstanceError):
        v.x = 99


def test_slots_no_dict():
    # slots -> no per-instance __dict__
    assert not hasattr(Vec3(1, 2, 3), "__dict__")


def test_hashable():
    # frozen -> hashable, so a Vec3 can key a dict / live in a set safely
    # (a mutable key would be a bug
    assert {Vec3(1, 2, 3): "a"}[Vec3(1, 2, 3)] == "a"
