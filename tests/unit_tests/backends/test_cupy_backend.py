"""CuPy backend contract (gpu).

The CPU backend + ``get_backend`` dispatch live in ``test_base.py``; this file
holds the GPU-only pieces, which ``importorskip`` away when CuPy / CUDA is
absent.
"""

import numpy as np
import pytest


@pytest.fixture
def cupy_backend():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("No CUDA device")
    from fast_heat_solv.backends import get_backend
    return get_backend("cupy")


@pytest.mark.gpu
def test_get_backend_cupy(cupy_backend):
    import cupy

    from fast_heat_solv.backends.cupy_backend import CupyBackend
    from fast_heat_solv.physics import spectral_gpu_kernels

    assert isinstance(cupy_backend, CupyBackend)
    assert cupy_backend.name == "cupy"
    assert cupy_backend.xp is cupy
    assert cupy_backend.kernels is spectral_gpu_kernels


@pytest.mark.gpu
def test_to_numpy_device_get(cupy_backend):
    import cupy

    host = np.arange(12, dtype=np.float32).reshape(3, 4)
    device = cupy.asarray(host)
    out = cupy_backend.to_numpy(device)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, host)


def test_get_backend_cupy_importerror_without_cupy():
    """When CuPy is genuinely absent, get_backend('cupy') raises ImportError."""
    import importlib.util

    if importlib.util.find_spec("cupy") is not None:
        pytest.skip("CuPy is installed; the absent-CuPy path can't be exercised")
    from fast_heat_solv.backends import get_backend

    with pytest.raises(ImportError):
        get_backend("cupy")
