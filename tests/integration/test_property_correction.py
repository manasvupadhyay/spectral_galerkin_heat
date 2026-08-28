"""Integration gates for the temperature-dependent property correction.

CPU-side checks that must pass before any GPU / FE-reference comparison:

* null test — constant branches equal to the reference reproduce the base
  (constant-coefficient) solver bit-for-bit;
* stability / physicality — the correction keeps the field finite, positive
  properties, surface bounded, and the Picard loop converging;
* convergence parameter eps stays in the modest regime.

Run with::

    pytest -m integration tests/integration/test_property_correction.py

Heavy solver imports are deferred into the test bodies.
"""

import copy

import numpy as np
import pytest

from fast_heat_solv.core.parameters import SimulationContext

# 316L typical T-dependent properties
_K_BR = {"solid": "9.248 + 0.01571 * T", "liquid": "12.41 + 0.003279 * T",
         "reference": 24.5075, "unit": "W/(m.K)"}
_RHO_BR = {"solid": "8084.2 - 0.42086 * T - 3.8942e-5 * T**2",
           "liquid": "7432.7 + 0.039338 * T - 1.8007e-4 * T**2",
           "reference": 6896.2365, "unit": "kg/m^3"}
_CP_BR = {"solid": "458.98 + 0.1328 * T", "liquid": "769.86",
          "reference": 633.8752, "unit": "J/(kg.K)"}

_T_S, _T_L, _T0 = 1674.15, 1697.15, 293.0
_LX, _LY, _LZ = 1.0e-3, 0.5e-3, 0.25e-3
_A, _P, _R_B, _V = 0.30, 24.0, 30e-6, 0.15
_X_START = _LX / 4
_T_TOTAL = 1.0e-3


def _cfg(material, mesh=(48, 24, 32), n_steps=20, backend="cpu", fine=None,
         dtype=None):
    cfg = {
        "simulation": {"backend": backend,
                       "duration": _T_TOTAL, "dt": _T_TOTAL / n_steps},
        "domain": {"size": [_LX, _LY, _LZ], "mesh": list(mesh)},
        "material": material,
        "laser": {"radius": _R_B, "absorptivity": _A, "power_nominal": _P},
        "io": {},
    }
    if fine is not None:
        cfg["fine_mesh"] = fine
    if dtype is not None:
        cfg["simulation"]["dtype"] = dtype
    return cfg


def _base_material(**over):
    mat = {"name": "316L", "T_solidus": _T_S, "T_liquidus": _T_L, "T0": _T0,
           "L_f": 2.677e5, "DeltaH_LV": 7.416e6, "R_v": 150.774,
           "Pa": 101325.0, "T_boil": 3090.0}
    mat.update(over)
    return mat


def _context(cfg, laser):
    ctx = SimulationContext.from_dict(copy.deepcopy(cfg))
    ctx.laser_path = laser
    return ctx


def _run(ctx, max_picard_iter=None, track_history=False):
    from fast_heat_solv.physics.spectral_helpers import reconstruct_temperature_DCT
    from fast_heat_solv.solvers import build_solver

    solver = build_solver(ctx)
    solver.initialize(ctx)
    if max_picard_iter is not None:
        solver.max_picard_iter = max_picard_iter
    solver.track_picard_history = track_history
    dt = ctx.num.dt
    surf_max, picard = [], []
    last_history = None
    for step in range(ctx.num.n_steps):
        _, metrics = solver.step(step * dt, dt)
        surf_max.append(float(metrics["T_surface_max"]))
        picard.append(int(metrics["n_evap_iter"]))
        if track_history:
            last_history = list(solver.picard_history)
    T = reconstruct_temperature_DCT(solver.state.a, solver.state)
    if track_history:
        return solver, T, surf_max, picard, last_history
    return solver, T, surf_max, picard


# ---------------------------------------------------------------------------
# null test — constant branches == base solver, bit-for-bit
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_null_constant_branches_match_scalar(constant_velocity_laser):
    """Constant property branches (= reference) reproduce the scalar solver exactly.

    The model is flagged constant, so the correction path is disabled — the run
    must be identical to the equivalent scalar-material run to the bit.
    """
    k0, rho0, cp0 = 15.0, 7900.0, 500.0
    scalar = _cfg(_base_material(k=k0, rho=rho0, Cp=cp0))
    const_branch = _cfg(_base_material(
        k={"solid": str(k0), "liquid": str(k0)},
        rho={"solid": str(rho0), "liquid": str(rho0)},
        Cp={"solid": str(cp0), "liquid": str(cp0)},
    ))

    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    _, T_scalar, _, _ = _run(_context(scalar, laser))
    laser2 = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    solver_b, T_branch, _, _ = _run(_context(const_branch, laser2))

    assert solver_b._property_correction is False
    np.testing.assert_array_equal(T_scalar, T_branch)


# ---------------------------------------------------------------------------
# stability / physicality with the correction active
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_temperature_dependent_run_is_physical(constant_velocity_laser):
    """T-dependent 316L run stays finite, physical, bounded, and converges.

    Uses a fine box (box mode), as every production T-dependent config does: the
    correction resolves a sharp mushy-zone source that the coarse main grid
    (grid mode) under-resolves into ringing at this resolution.
    """
    cfg = _cfg(_base_material(k=_K_BR, rho=_RHO_BR, Cp=_CP_BR),
               fine={"refinement": 2, "box_size": [0.4e-3, 0.3e-3, 0.06e-3]})
    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    ctx = _context(cfg, laser)

    solver, T, surf_max, picard, history = _run(
        ctx, max_picard_iter=80, track_history=True)
    assert solver._property_correction is True

    # Finite, surface bounded by the evaporation cap.
    assert not np.isnan(T).any() and not np.isinf(T).any()
    assert max(surf_max) < 1.5 * 3090.0
    peak_excursion = float(T.max()) - _T0
    assert T.min() >= _T0 - 0.05 * peak_excursion, (
        f"undershoot {(_T0 - float(T.min())):.1f} K > 5% of peak "
        f"{peak_excursion:.1f} K — looks like divergence, not ringing")

    # Properties stay positive over the whole realised temperature range
    # (the reason for using tabulated fluctuations, not a linearisation).
    Trange = np.linspace(_T0, max(surf_max), 200)
    assert float(np.min(ctx.mat.model.k(Trange))) > 0.0
    assert float(np.min(ctx.mat.model.a(Trange))) > 0.0

    # Convergence is *not impaired*: the final step converges below the cap and
    # its residual decreases monotonically (stable fixed point, all-orders
    # resummation).
    assert picard[-1] < solver.max_picard_iter
    rms = [h["rms_diff"] for h in history]
    assert all(b <= a for a, b in zip(rms, rms[1:])), "residual not monotone"


# ---------------------------------------------------------------------------
# convergence parameter ε ~ ||k'|| / (k̄ sqrt(V))
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_epsilon_in_safe_regime(constant_velocity_laser):
    """The controlling parameter ε stays modest, so the series converges."""
    from fast_heat_solv.physics.spectral_ops import reconstruct_volume

    # This short track stays cold in the bulk, so the reference that keeps the
    # fluctuation k'=k(T)-k̄ modest is the cold-side value k(T0) (= the 316L solid
    # branch at 293 K), not the [T0, T_boil] mid-range. Reference the branches at
    # T0 to exercise the safe-regime guardrail.
    ref = {"k": 13.851, "rho": 7957.5, "Cp": 497.89}
    cfg = _cfg(_base_material(
        k={**_K_BR, "reference": ref["k"]},
        rho={**_RHO_BR, "reference": ref["rho"]},
        Cp={**_CP_BR, "reference": ref["Cp"]}))
    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    ctx = _context(cfg, laser)
    solver, _, _, _ = _run(ctx)

    T = reconstruct_volume(solver.state.a, solver.state)
    k_prime = ctx.mat.model.k(T) - float(solver._kbar)
    eps = float(np.sqrt(np.mean((k_prime / float(solver._kbar)) ** 2)))
    assert eps < 0.3, f"ε={eps:.4f} outside the modest regime — series may diverge"


# ---------------------------------------------------------------------------
# precision — the T-dependent path must honour simulation.dtype
# ---------------------------------------------------------------------------
#
# The property kernels are generated source (numba on CPU, ElementwiseKernel on
# GPU) emitting the branch expressions and their literals directly, so precision
# is a codegen decision rather than an array-dtype one. Nothing else in the suite
# exercises dtype together with T-dependent properties: the other dtype tests
# use scalar k/rho/Cp, which never reach this path.

_TDEP_FINE = {"refinement": 2, "box_size": [0.4e-3, 0.3e-3, 0.06e-3]}


def _tdep_run(backend, dtype, laser, n_steps=6, cap=80):
    """Run a T-dependent case at one precision; return (solver, T, picard).
    """
    cfg = _cfg(_base_material(k=_K_BR, rho=_RHO_BR, Cp=_CP_BR),
               n_steps=n_steps, backend=backend, fine=_TDEP_FINE, dtype=dtype)
    solver, T, surf_max, picard = _run(_context(cfg, laser), max_picard_iter=cap)
    assert solver._property_correction is True
    return solver, T, picard


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("dtype_name, np_dtype", [
    ("float32", np.float32),
    ("float64", np.float64),
])
def test_tdep_cpu_honours_dtype(constant_velocity_laser, dtype_name, np_dtype):
    """The configured precision reaches the generated property kernels."""
    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    solver, T, _ = _tdep_run("cpu", dtype_name, laser)

    assert solver.state.dtype == np_dtype
    assert solver.state.a.dtype == np.dtype(np_dtype)
    assert T.dtype == np.dtype(np_dtype)
    assert solver._T_prev_full.dtype == np.dtype(np_dtype)
    assert not np.isnan(T).any() and not np.isinf(T).any()
    assert T.max() > _T0

    # The generated kernels are cached per precision, not per model.
    model = solver.context.mat.model
    assert np.dtype(np_dtype) in model._kp_ap_cpu_kernel


@pytest.mark.integration
@pytest.mark.slow
def test_tdep_cpu_float64_matches_float32(constant_velocity_laser):
    """Both precisions solve the same problem, so the fields must agree.

    float32 sets the error floor; the bound is a small fraction of the peak
    excursion.
    """
    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    s32, T32, p32 = _tdep_run("cpu", "float32", laser, n_steps=20)
    s64, T64, p64 = _tdep_run("cpu", "float64", laser, n_steps=20)

    assert p32[-1] < s32.max_picard_iter and p64[-1] < s64.max_picard_iter, (
        f"a run bailed at the Picard cap (f32={p32[-1]}, f64={p64[-1]}, "
        f"cap={s32.max_picard_iter}); the comparison below would be meaningless")

    assert T32.shape == T64.shape
    excursion = float(T64.max()) - _T0
    max_abs_diff = float(np.max(np.abs(T64 - T32.astype(np.float64))))
    assert max_abs_diff <= 1e-3 * excursion, (
        f"float32 and float64 T-dependent runs differ by {max_abs_diff:.3e} K, "
        f"exceeding 1e-3 * peak excursion = {1e-3 * excursion:.3e} K")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("dtype_name, np_dtype", [
    ("float32", np.float32),
    ("float64", np.float64),
])
def test_tdep_gpu_honours_dtype(constant_velocity_laser, dtype_name, np_dtype):
    """The configured precision reaches the generated property kernels.
    """
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CuPy not available or no CUDA device found")

    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    solver, T, _ = _tdep_run("gpu", dtype_name, laser)

    assert solver.state.a.dtype == np.dtype(np_dtype)
    T_host = cupy.asnumpy(T) if hasattr(T, "get") else T
    assert T_host.dtype == np.dtype(np_dtype)
    assert not np.isnan(T_host).any() and not np.isinf(T_host).any()
    assert T_host.max() > _T0

    model = solver.context.mat.model
    assert np.dtype(np_dtype) in model._kp_ap_kernel


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
def test_tdep_cpu_gpu_float64_equivalence(constant_velocity_laser):
    """CPU and GPU must agree in double precision on the T-dependent path.
    """
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CuPy not available or no CUDA device found")

    laser = constant_velocity_laser(_X_START, _LY / 2, _V, 0.0, _P)
    s_cpu, T_cpu, p_cpu = _tdep_run("cpu", "float64", laser, n_steps=20)
    s_gpu, T_gpu, p_gpu = _tdep_run("gpu", "float64", laser, n_steps=20)
    T_gpu = cupy.asnumpy(T_gpu) if hasattr(T_gpu, "get") else T_gpu

    assert p_cpu[-1] < s_cpu.max_picard_iter and p_gpu[-1] < s_gpu.max_picard_iter, (
        f"a run bailed at the Picard cap (cpu={p_cpu[-1]}, gpu={p_gpu[-1]})")
    assert T_cpu.shape == T_gpu.shape
    excursion = float(T_cpu.max()) - _T0
    max_abs_diff = float(np.max(np.abs(T_gpu - T_cpu)))
    assert max_abs_diff <= 1e-4 * excursion, (
        f"CPU and GPU float64 T-dependent fields differ by {max_abs_diff:.3e} K, "
        f"exceeding 1e-4 * peak excursion = {1e-4 * excursion:.3e} K")
