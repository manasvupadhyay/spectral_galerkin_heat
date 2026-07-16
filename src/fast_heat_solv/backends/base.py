"""Math backend container — bundles an array module with its physics kernels."""

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

import numpy as _np


def to_host(arr):
    """Return a host :class:`numpy.ndarray` for *arr* (NumPy or CuPy).

    No-op (a plain ``numpy.asarray``) when *arr* is already a host array;
    calls ``arr.get()`` when *arr* is a CuPy device array.
    """
    get = getattr(arr, "get", None)
    return get() if callable(get) else _np.asarray(arr)


class MathBackend:
    """Bundles the array module and physics kernels for one execution target.

    NumPy and CuPy expose the same API, so a backend is a thin container: the
    solver does ``xp = backend.xp`` and then calls ``xp.zeros(...)``,
    ``xp.multiply(...)``, etc. directly. Host-transfer is the only operation
    that differs between targets, so :meth:`to_numpy` is the single method.

    Parameters
    ----------
    name : str
        Short identifier of the backend, ``"numpy"`` or ``"cupy"``.
    xp : module
        The array module — :mod:`numpy` or :mod:`cupy`.
    kernels : module
        The physics kernel module matching *xp*
        (``spectral_cpu_kernels`` or ``spectral_gpu_kernels``).
    """

    def __init__(self, name: str, xp, kernels):
        self.name = name
        self.xp = xp
        self.kernels = kernels

    def to_numpy(self, arr):
        """Return a host :class:`numpy.ndarray` for *arr*.

        No-op (a plain ``numpy.asarray``) when *arr* is already a host array;
        calls ``arr.get()`` when *arr* is a CuPy device array.

        Parameters
        ----------
        arr : ndarray
            A NumPy or CuPy array.

        Returns
        -------
        numpy.ndarray
            The array on the host.
        """
        return to_host(arr)

    def __repr__(self) -> str:
        return f"MathBackend(name={self.name!r})"
