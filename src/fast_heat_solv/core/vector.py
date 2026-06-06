"""Tiny immutable 3-component value type for coordinate triples.

``Vec3`` groups the ``(x, y, z)`` scalar triples that recur throughout the
geometry and state-construction code — domain size, mesh counts, grid spacing —
so they travel as one named object instead of three loose scalars.

It deliberately lives at the **Python orchestration layer only**: numba
``@njit`` / ``@cuda.jit`` kernels cannot accept a dataclass, so values are still
unpacked to plain scalars/arrays at the kernel boundary. The payoff is
readability and a single, explicit home for the ``[z, y, x]`` array-index
ordering convention via :meth:`Vec3.zyx`.
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

from dataclasses import dataclass
from typing import Callable, Iterator, Union

Number = Union[int, float]


@dataclass(frozen=True, slots=True)
class Vec3:
    """An immutable ``(x, y, z)`` triple of scalars.

    Iterable (so ``a, b, c = v`` and ``zip(v, w)`` work) and supports
    element-wise division for the common ``size / n -> spacing`` idiom.

    Attributes
    ----------
    x, y, z : int or float
        The three components, in natural (x, y, z) order.

    Examples
    --------
    >>> size = Vec3(4.0, 2.0, 1.0)
    >>> n = Vec3(8, 4, 2)
    >>> tuple(size / n)
    (0.5, 0.5, 0.5)
    >>> tuple(n.zyx())
    (2, 4, 8)
    """

    x: Number
    y: Number
    z: Number

    def __iter__(self) -> Iterator[Number]:
        """Yield the components in (x, y, z) order for unpacking / ``zip``."""
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, i: int) -> Number:
        """Index the components as ``0 -> x``, ``1 -> y``, ``2 -> z``."""
        return (self.x, self.y, self.z)[i]

    def zyx(self) -> "Vec3":
        """Return the triple reversed to ``(z, y, x)`` — array-index order.

        Arrays in the solver are indexed ``[z, y, x]``; call this at the few
        sites that build per-axis quantities in array order so the reversal is
        explicit and greppable instead of an implicit literal.
        """
        return Vec3(self.z, self.y, self.x)

    def map(self, f: Callable[[Number], Number]) -> "Vec3":
        """Return a new ``Vec3`` with *f* applied to each component."""
        return Vec3(f(self.x), f(self.y), f(self.z))

    def __truediv__(self, other: Union["Vec3", Number]) -> "Vec3":
        """Element-wise division by another ``Vec3`` or a scalar."""
        o = other if isinstance(other, Vec3) else Vec3(other, other, other)
        return Vec3(self.x / o.x, self.y / o.y, self.z / o.z)

    def __mul__(self, other: Union["Vec3", Number]) -> "Vec3":
        """Element-wise multiplication by another ``Vec3`` or a scalar."""
        o = other if isinstance(other, Vec3) else Vec3(other, other, other)
        return Vec3(self.x * o.x, self.y * o.y, self.z * o.z)
