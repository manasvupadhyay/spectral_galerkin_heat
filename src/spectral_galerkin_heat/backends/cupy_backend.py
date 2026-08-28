"""GPU math backend: CuPy arrays with Numba CUDA kernels.

This module imports :mod:`cupy`, so it is imported lazily (only from
:func:`spectral_galerkin_heat.backends.get_backend`) to keep CuPy an optional
dependency.
"""

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

import cupy as cp

import spectral_galerkin_heat.physics.spectral_gpu_kernels as _gpu_kernels

from .base import MathBackend


class CupyBackend(MathBackend):
    """GPU backend: CuPy arrays and Numba CUDA kernels."""

    def __init__(self):
        super().__init__(name="cupy", xp=cp, kernels=_gpu_kernels)
