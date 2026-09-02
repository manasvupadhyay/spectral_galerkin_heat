"""Tests for the CPU numba kernels & transforms (physics/spectral_cpu_kernels.py).

The numba/pyfftw calls JIT on first use (cached to disk after), hence `slow`.
Physics-formula assertions re-derive the documented formula independently as the
oracle; the physical constants have been reviewed and signed off.
"""

import numpy as np
import pytest

from spectral_galerkin_heat.core.parameters import GeomParams
from spectral_galerkin_heat.core.vector import Vec3
from spectral_galerkin_heat.physics import spectral_cpu_kernels as k

# --- Transforms -------------------------------------------------------------

@pytest.mark.slow
def test_dct_idct_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 4, 4)).astype(np.float32)
    np.testing.assert_allclose(k.IDCT_II(k.DCT_II(x)), x, rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_dct_of_constant_is_single_impulse():
    # Ortho DCT-II of a constant field has energy only in the (0,0,0) mode;
    # magnitude = prod(sqrt(n_axis)) = n^{3/2} for an n^3 cube of ones.
    n = 4
    out = k.DCT_II(np.ones((n, n, n), dtype=np.float32))
    assert out[0, 0, 0] == pytest.approx(n ** 1.5, rel=1e-4)
    out[0, 0, 0] = 0.0
    assert np.allclose(out, 0.0, atol=1e-4)


# --- ETD mode updates (hand-computed) ---------------------------------------

@pytest.mark.slow
def test_update_modes_etd1():
    # a_out[p] = aK[p] + KK[p] * Cp[p] * B_scaled
    aK = np.ones((2, 2, 2), dtype=np.float32)
    KK = np.ones((2, 2, 2), dtype=np.float32)
    Cp = np.array([2.0, 3.0], dtype=np.float32).reshape(2, 1, 1)
    B = np.full((2, 2), 5.0, dtype=np.float32)
    out = np.empty((2, 2, 2), dtype=np.float32)
    k.update_modes_etd1(aK, KK, Cp, B, out)
    assert np.allclose(out[0], 1 + 1 * 2 * 5)
    assert np.allclose(out[1], 1 + 1 * 3 * 5)


@pytest.mark.slow
def test_add_source_term_modes():
    a = np.ones((2, 2, 2), dtype=np.float32)
    KK = np.full((2, 2, 2), 2.0, dtype=np.float32)
    Q = np.full((2, 2, 2), 3.0, dtype=np.float32)
    k.add_source_term_modes(a, KK, Q)
    assert np.allclose(a, 1 + 2 * 3)  # a += KK*Q


@pytest.mark.slow
def test_add_bottom_surface_source():
    # a[p] += KK[p] * Cp_bottom[p] * B  (B is the 2-D z=0 forcing, broadcast over modes p)
    a = np.ones((2, 2, 2), dtype=np.float32)
    KK = np.ones((2, 2, 2), dtype=np.float32)
    Cp_bot = np.array([2.0, 4.0], dtype=np.float32).reshape(2, 1, 1)
    B = np.full((2, 2), 5.0, dtype=np.float32)
    k.add_bottom_surface_source(a, KK, Cp_bot, B)
    assert np.allclose(a[0], 1 + 1 * 2 * 5)  # 1 + KK*Cp_bottom[0]*B = 1 + 1*2*5 = 11
    assert np.allclose(a[1], 1 + 1 * 4 * 5)  # 1 + 1*4*5 = 21


# --- Source terms -----------------------------------------------------------

def test_gaussian_laser_flux_peak_and_integral():
    # Plain NumPy. Peak at the beam centre == laser_coef; the discrete
    # integral matches the analytic Gaussian integral laser_coef * pi*r^2/2.
    nx = 200
    dx = 5e-6
    coords = (np.arange(nx) + 0.5) * dx
    cx = cy = coords[nx // 2]
    r, coef = 1e-4, 7.0
    q = k.compute_gaussian_laser_flux(coords, coords, cx, cy, r, coef)
    assert q.max() == pytest.approx(coef, rel=1e-6)
    integral = q.sum() * dx * dx
    assert integral == pytest.approx(coef * np.pi * r**2 / 2, rel=0.02)

def test_shift_flux_half_pixel_interpolates():
    # A sub-pixel shift exercises the order-1 (linear) interpolation, not just
    # integer translation. Use a ramp in x so the interpolated value is exact:
    # a +0.5 px shift averages neighbours -> out[i] = i - 0.5 (interior).
    geom = GeomParams(size=Vec3(8.0, 8.0, 1.0), n=Vec3(8, 8, 4))  # d = (1, 1, ..)
    field = np.tile(np.arange(8, dtype=np.float32), (8, 1))  # field[j, i] = i
    out = k.shift_flux(field, (0.5, 0.0), geom)  # +0.5 px in x
    assert out[4, 4] == pytest.approx(3.5, rel=1e-5)  # 0.5*field[4,3] + 0.5*field[4,4]
    assert out[4, 5] == pytest.approx(4.5, rel=1e-5)


# --- Latent heat (mushy zone) ----------------------------------------------

@pytest.mark.slow
def test_latent_heat_source_indicator_and_sign():
    # Oracle = the documented formula Q = -rho*L/((T_L-T_S)*dt) * dT,
    # active only inside the mushy band [T_S, T_L]. (physics confirmed)
    T_S, T_L, rho, L, dt = 1700.0, 1800.0, 7900.0, 2.5e5, 1e-6
    T_curr = np.array([[[1600.0, 1750.0, 1900.0]]], dtype=np.float32)  # below / in / above
    T_prev = np.array([[[1600.0, 1700.0, 1900.0]]], dtype=np.float32)
    out = np.empty_like(T_curr)
    k.compute_source_term_from_temperature(T_curr, T_prev, T_S, T_L, rho, L, dt, out)
    assert out[0, 0, 0] == 0.0   # below band
    assert out[0, 0, 2] == 0.0   # above band
    # in band, heating (dT>0) absorbs energy -> negative source of exact magnitude.
    expected = -rho * L / ((T_L - T_S) * dt) * (1750.0 - 1700.0)
    assert out[0, 0, 1] == pytest.approx(expected, rel=1e-4)


@pytest.mark.slow
def test_latent_heat_clamps_tprev_into_band():
    # T_prev is clamped into [T_S, T_L] before differencing (mechanical).
    T_S, T_L, rho, L, dt = 1700.0, 1800.0, 7900.0, 2.5e5, 1e-6
    T_curr = np.array([[[1750.0]]], dtype=np.float32)
    T_prev = np.array([[[1000.0]]], dtype=np.float32)  # below T_S -> clamped to T_S
    out = np.empty_like(T_curr)
    k.compute_source_term_from_temperature(T_curr, T_prev, T_S, T_L, rho, L, dt, out)
    expected = -rho * L / ((T_L - T_S) * dt) * (1750.0 - T_S)
    assert out[0, 0, 0] == pytest.approx(expected, rel=1e-4)


# --- Evaporation flux -------------------------------------------------------

@pytest.mark.slow
def test_evaporation_arrhenius_form_and_monotonic():
    # Above liquidus the flux follows the documented Arrhenius law; oracle =
    # the independently re-derived formula (physics confirmed).
    P0, T_boil, dH, R_v, T_liq = 101325.0, 3090.0, 7.41e6, 150.774, 1800.0
    factor1 = 0.82 * dH * P0 / np.sqrt(2 * np.pi * R_v)
    factor2 = dH / (R_v * T_boil)

    def expected(T):
        return factor1 * (1.0 / np.sqrt(T)) * np.exp(factor2 * (1.0 - T_boil / T))

    T = np.array([[2000.0, 2200.0]], dtype=np.float32)
    q = np.empty((1, 2), dtype=np.float32)
    k.compute_evaporation_flux(T, q, P0, T_boil, dH, R_v, T_liq)
    assert q[0, 0] > 0.0                                       # non-zero above liquidus
    assert q[0, 0] == pytest.approx(expected(2000.0), rel=1e-4)  # exact formula
    assert q[0, 1] > q[0, 0]                                   # rises with surface T
