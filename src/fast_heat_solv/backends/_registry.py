"""Backend registry — map names to backend factories and resolve them.

The registry keeps the *catalog* of available backends separate from the
solver code that consumes them. New backends register themselves with the
:func:`register_backend` decorator (or by inserting a factory directly), so
adding one never requires editing :func:`get_backend`.
"""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique
#
# Author: Théo Andrieux
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__author__ = "Théo Andrieux"
__copyright__ = "Copyright 2026, LMS, École Polytechnique"

from collections.abc import Callable
from typing import TypeVar

from .base import MathBackend

# Maps a backend name to a zero-argument factory returning a fresh instance.
# A *factory* (rather than a class) is stored so that optional backends can
# defer their imports until requested (see the lazy CuPy registration in
# ``backends/__init__.py``).
_BACKEND_FACTORIES: dict[str, Callable[[], MathBackend]] = {}

F = TypeVar("F", bound=Callable[[], MathBackend])


def register_backend(name: str) -> Callable[[F], F]:
    """Register a backend factory under *name*.

    Use as a decorator on a :class:`MathBackend` subclass (the class itself is
    a zero-argument factory) or on a plain factory function::

        @register_backend("numpy")
        class NumpyBackend(MathBackend):
            ...

    Parameters
    ----------
    name : str
        Key under which the backend is looked up by :func:`get_backend`.

    Returns
    -------
    Callable
        A decorator that registers its argument and returns it unchanged.
    """

    def decorator(factory: F) -> F:
        _BACKEND_FACTORIES[name] = factory
        return factory

    return decorator


def get_backend(name: str = "numpy") -> MathBackend:
    """Return a :class:`MathBackend` by registered name.

    Parameters
    ----------
    name : str, optional
        Name of a registered backend, e.g. ``"numpy"`` for the CPU backend or
        ``"cupy"`` for the GPU backend. Defaults to ``"numpy"``.

    Returns
    -------
    MathBackend
        A fresh instance of the requested backend.

    Raises
    ------
    ValueError
        If no backend is registered under *name*.
    ImportError
        If the backend is registered but its optional dependency (e.g. CuPy)
        is not installed.
    """
    factory = _BACKEND_FACTORIES.get(name)
    if factory is None:
        available = ", ".join(sorted(_BACKEND_FACTORIES)) or "(none)"
        raise ValueError(
            f"Unknown backend: {name!r}. Registered backends: {available}."
        )
    return factory()
