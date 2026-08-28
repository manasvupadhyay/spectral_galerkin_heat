"""Container pairing an array module with its physics kernels."""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS),
# École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris,
# Route de Saclay, Palaiseau, 91128, France.
#
# Author: Théo Andrieux, Jules Dichamp, Manas V. Upadhyay
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


__author__ = "Théo Andrieux, Jules Dichamp, Manas V. Upadhyay"
__copyright__ = "Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris"

import numpy as _np


def to_host(arr):
    """Return *arr* as a :class:`numpy.ndarray` on the host.

    Calls ``numpy.asarray`` if *arr* is already a host array, or ``arr.get()``
    if it is a CuPy device array.
    """
    get = getattr(arr, "get", None)
    return get() if callable(get) else _np.asarray(arr)


class MathBackend:
    """Hold the array module and physics kernels for one execution target.

    NumPy and CuPy share the same API, so the solver sets ``xp = backend.xp``
    and calls ``xp.zeros(...)`` directly. Host transfer is the only operation
    that differs between the two, which is why :meth:`to_numpy` is the only
    method.

    Parameters
    ----------
    name : str
        Backend identifier, ``"numpy"`` or ``"cupy"``.
    xp : module
        :mod:`numpy` or :mod:`cupy`.
    kernels : module
        Kernel module matching *xp* (``spectral_cpu_kernels`` or
        ``spectral_gpu_kernels``).
    """

    def __init__(self, name: str, xp, kernels):
        self.name = name
        self.xp = xp
        self.kernels = kernels

    def to_numpy(self, arr):
        """Return *arr* as a :class:`numpy.ndarray` on the host.

        Calls ``numpy.asarray`` if *arr* is already a host array, or
        ``arr.get()`` if it is a CuPy device array.

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
