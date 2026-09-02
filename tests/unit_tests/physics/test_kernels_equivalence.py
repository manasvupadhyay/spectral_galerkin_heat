"""CPU↔GPU kernel parity.

For each numerical kernel that exists in both ``spectral_cpu_kernels`` and
``spectral_gpu_kernels``, feed *identical* inputs (host array → device for the
GPU call) and assert the results agree within float tolerance.  Also guards that
the two modules expose the same public ``__all__`` 

Skipped cleanly when CuPy / CUDA is unavailable.
"""

import inspect

import numpy as np
import pytest

from spectral_galerkin_heat.physics import spectral_cpu_kernels as cpu


@pytest.fixture
def gpu():
    cupy = pytest.importorskip("cupy")
    try:
        _ = cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("No CUDA device")
    from spectral_galerkin_heat.physics import spectral_gpu_kernels as gpu_mod
    return cupy, gpu_mod


# --- contract: the two modules must not drift apart -------------------------

def test_public_api_matches():
    pytest.importorskip("cupy")  # importing the GPU module needs cupy + numba.cuda
    from spectral_galerkin_heat.physics import spectral_gpu_kernels as gpu_mod
    assert cpu.__all__ == gpu_mod.__all__


def test_evaporation_signature_matches():
    pytest.importorskip("cupy")
    from spectral_galerkin_heat.physics import spectral_gpu_kernels as gpu_mod
    sig_cpu = inspect.signature(cpu.compute_evaporation_flux)
    sig_gpu = inspect.signature(gpu_mod.compute_evaporation_flux)
    assert list(sig_cpu.parameters) == list(sig_gpu.parameters)


# --- numerical parity -------------------------------------------------------

RTOL = 1e-5  # absorbs CPU/GPU FFT and reduction-order differences (float32)


@pytest.mark.gpu
def test_dct_idct_parity(gpu):
    cupy, gpu_mod = gpu
    rng = np.random.default_rng(0)
    q = rng.standard_normal((8, 8, 8)).astype(np.float32)
    np.testing.assert_allclose(cupy.asnumpy(gpu_mod.DCT_II(cupy.asarray(q))),
                               cpu.DCT_II(q), rtol=RTOL, atol=1e-4)
    a = rng.standard_normal((8, 8, 8)).astype(np.float32)
    np.testing.assert_allclose(cupy.asnumpy(gpu_mod.IDCT_II(cupy.asarray(a))),
                               cpu.IDCT_II(a), rtol=RTOL, atol=1e-4)


@pytest.mark.gpu
def test_gaussian_laser_flux_parity(gpu):
    cupy, gpu_mod = gpu
    x = np.linspace(0, 4e-4, 32, dtype=np.float32)
    y = np.linspace(0, 4e-4, 24, dtype=np.float32)
    args = (2e-4, 2e-4, 60e-6, 1.0e11)
    out_cpu = cpu.compute_gaussian_laser_flux(x, y, *args)
    out_gpu = gpu_mod.compute_gaussian_laser_flux(cupy.asarray(x), cupy.asarray(y), *args)
    np.testing.assert_allclose(cupy.asnumpy(out_gpu), out_cpu, rtol=RTOL)


@pytest.mark.gpu
def test_evaporation_flux_parity(gpu):
    cupy, gpu_mod = gpu
    T_surf = np.linspace(1500.0, 3500.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    consts = {"P0": 101325.0, "T_boil": 3090.0, "DeltaH_LV": 7.41e6,
              "R_v": 150.774, "T_liquidus": 1800.0}
    q_cpu = np.zeros_like(T_surf)
    cpu.compute_evaporation_flux(T_surf, q_cpu, *consts.values())
    q_gpu = cupy.zeros_like(cupy.asarray(T_surf))
    gpu_mod.compute_evaporation_flux(cupy.asarray(T_surf), q_gpu, *consts.values())
    np.testing.assert_allclose(cupy.asnumpy(q_gpu), q_cpu, rtol=RTOL)


@pytest.mark.gpu
def test_source_term_from_temperature_parity(gpu):
    cupy, gpu_mod = gpu
    rng = np.random.default_rng(1)
    # Temperatures straddling the mushy band so the indicator is exercised.
    T_curr = rng.uniform(1600.0, 1900.0, (8, 16, 16)).astype(np.float32)
    T_prev = rng.uniform(1600.0, 1900.0, (8, 16, 16)).astype(np.float32)
    args = (1700.0, 1800.0, 7850.0, 267700.0, 6e-6)  # T_S, T_L, rho, L, dt
    out_cpu = np.zeros_like(T_curr)
    cpu.compute_source_term_from_temperature(T_curr, T_prev, *args, out_cpu)
    out_gpu = cupy.zeros_like(cupy.asarray(T_curr))
    gpu_mod.compute_source_term_from_temperature(
        cupy.asarray(T_curr), cupy.asarray(T_prev), *args, out_gpu)
    np.testing.assert_allclose(cupy.asnumpy(out_gpu), out_cpu, rtol=RTOL)


@pytest.mark.gpu
def test_update_modes_etd1_parity(gpu):
    cupy, gpu_mod = gpu
    rng = np.random.default_rng(2)
    aK = rng.standard_normal((8, 8, 8)).astype(np.float32)
    KK = rng.standard_normal((8, 8, 8)).astype(np.float32)
    Cp = rng.standard_normal((8, 1, 1)).astype(np.float32)
    B = rng.standard_normal((8, 8)).astype(np.float32)
    out_cpu = np.empty_like(aK)
    cpu.update_modes_etd1(aK, KK, Cp, B, out_cpu)
    out_gpu = cupy.empty_like(cupy.asarray(aK))
    gpu_mod.update_modes_etd1(cupy.asarray(aK), cupy.asarray(KK),
                              cupy.asarray(Cp), cupy.asarray(B), out_gpu)
    np.testing.assert_allclose(cupy.asnumpy(out_gpu), out_cpu, rtol=RTOL)
