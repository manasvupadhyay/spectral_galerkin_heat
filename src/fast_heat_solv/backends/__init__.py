"""Math backends — dispatch array operations and kernels to CPU or GPU.

A :class:`MathBackend` bundles an array module (NumPy or CuPy) with its
matching physics-kernel module. Inject one into
:class:`~fast_heat_solv.solvers.spectral.SpectralSolver` to run on CPU or GPU
from the same solver code.

Want to run on a different array library (PyTorch, JAX, …)? See "Adding a
backend" in ``docs/ARCHITECTURE.md`` 

Examples
--------
>>> from fast_heat_solv.backends import get_backend
>>> backend = get_backend("numpy")
>>> backend.name
'numpy'
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

from .base import MathBackend
from ._registry import get_backend, register_backend

# Importing the module runs its ``@register_backend`` decorator.
from .numpy_backend import NumpyBackend


@register_backend("cupy")
def _make_cupy_backend() -> MathBackend:
    # Registered as a lazy factory: CuPy is an optional dependency, so its
    # module is imported only when a CuPy backend is actually requested.
    from .cupy_backend import CupyBackend

    return CupyBackend()


__all__ = [
    "MathBackend",
    "NumpyBackend",
    "get_backend",
    "register_backend",
]
