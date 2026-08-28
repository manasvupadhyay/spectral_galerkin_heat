"""CPU math backend: NumPy arrays with Numba CPU kernels."""

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

import numpy as np

import spectral_galerkin_heat.physics.spectral_cpu_kernels as _cpu_kernels

from ._registry import register_backend
from .base import MathBackend


@register_backend("numpy")
class NumpyBackend(MathBackend):
    """CPU backend: NumPy arrays and Numba CPU kernels."""

    def __init__(self):
        super().__init__(name="numpy", xp=np, kernels=_cpu_kernels)
