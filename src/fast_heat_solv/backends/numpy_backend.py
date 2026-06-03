"""CPU math backend — NumPy arrays with Numba CPU kernels."""

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

import numpy as np

import fast_heat_solv.physics.spectral_cpu_kernels as _cpu_kernels
from .base import MathBackend


class NumpyBackend(MathBackend):
    """CPU backend: NumPy arrays and Numba CPU kernels."""

    def __init__(self):
        super().__init__(name="numpy", xp=np, kernels=_cpu_kernels)
