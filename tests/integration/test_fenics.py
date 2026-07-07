"""FEniCS / dolfinx finite-element cross-validation (§2.3, integration + fenics).

Cross-validates the spectral solver (SG) against an *independent* numerical
method — a P1 backward-Euler finite-element solve in dolfinx — in the **nonlinear
constant-property** regime: latent heat of fusion + Hertz-Knudsen evaporation +
bottom-face convection, with temperature-independent 316L properties (Chadwick
values at T0=293 K). This is **not** a golden file (the FE reference is recomputed 
each run). BEWARE the run takes ~25 min on a laptop CPU

(fine ``h_fine`` strip along the laser path, coarse elsewhere),
Newton-solved, with the Gaussian laser + evaporation on the top face and
convection on the bottom — the same physics the SG solver applies. 

Achievable agreement
--------------------
On this 400×400×100 µm domain with a ~150-step (600 µs) developed melt track,
SG-vs-FE agree to **L2_rel ≈ 0.88 %** (peaks match to ~0.15 %), computed by the
library's dedicated ``compute_L2_error.compare()`` on the FE mesh.

Dependency
-----------------
FEniCS/gmsh are heavy and not project dependencies — install via **conda only**
(no usable PyPI wheels for dolfinx), conda-forge channel::

    conda install -c conda-forge fenics-dolfinx=0.9.0 fenics-basix=0.9.0 \\
        fenics-ffcx=0.9.0 fenics-ufl=2024.2.0 mpi4py petsc4py python-gmsh

(matches ``andreas_heat_solv/environment.yaml``).  The test ``importorskip``s
them and is ``@pytest.mark.fenics`` so a CI FEniCS job can select it
(``pytest -m "integration and fenics"``).  It is heavy (~25 min: the nonlinear SG
solve dominates) — run only in the dedicated FEniCS job.
"""

import math

import numpy as np
import pytest

from fast_heat_solv.core.parameters import SimulationContext

# ---------------------------------------------------------------------------
# Shared physical case (constant-property 316L; nonlinearities on)
# ---------------------------------------------------------------------------
_LX, _LY, _LZ = 4e-4, 4e-4, 1e-4          # 400 × 400 × 100 µm 
_RHO, _K, _CP = 7958.0, 13.85, 498.0
_T_SOL, _T_LIQ, _L_F = 1674.15, 1697.15, 2.677e5
_DHLV, _T_BOIL, _PA, _RV = 7.416e6, 3090.0, 101325.0, 150.774
_H_CONV, _T_AMB = 3000.0, 293.0
_A, _P, _R_B, _V = 0.30, 24.0, 30e-6, 0.15
_X_START, _Y0 = _LX / 4, _LY / 2
_T0 = _T_AMB

_DT = 4e-6
_N_STEPS = 150                            # ~600 µs → a developed ~90 µm melt track
_H_FINE, _H_COARSE = 1e-6, 25e-6          # FE mesh: 1 µm fine strip, 25 µm coarse
_SG_NX, _SG_NY, _SG_NZ = 96, 96, 128

_MAT = {
    "name": "316L", "rho": _RHO, "k": _K, "Cp": _CP, "L_f": _L_F,
    "T_solidus": _T_SOL, "T_liquidus": _T_LIQ, "T0": _T0,
    "DeltaH_LV": _DHLV, "R_v": _RV, "Pa": _PA, "T_boil": _T_BOIL, "h_conv": _H_CONV,
}


def _laser_x(t):
    return _X_START + _V * t               # start-of-step sampling (matches SG ETD1)


# ---------------------------------------------------------------------------
# FE reference (dolfinx)
# ---------------------------------------------------------------------------

def _build_and_solve_fe():
    """Graded gmsh box + Newton-solved nonlinear FE; return (domain, T_solution)."""
    import gmsh
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.io.gmshio import model_to_mesh
    from dolfinx.nls.petsc import NewtonSolver
    from mpi4py import MPI
    from petsc4py import PETSc

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("box")
    occ = gmsh.model.occ
    vol = occ.add_box(0, 0, 0, _LX, _LY, _LZ)
    occ.synchronize()
    bnds = gmsh.model.get_boundary([(3, vol)], oriented=False, combined=False)
    tol = 1e-9
    top = [t for d, t in bnds if math.isclose(occ.get_center_of_mass(d, t)[2], _LZ, abs_tol=tol)]
    bot = [t for d, t in bnds if math.isclose(occ.get_center_of_mass(d, t)[2], 0.0, abs_tol=tol)]
    # Fine box localized to the laser trajectory + thermal trail (keeps 1 µm affordable).
    fld = gmsh.model.mesh.field
    fld.add("Box", 1)
    fld.setNumber(1, "VIn", _H_FINE)
    fld.setNumber(1, "VOut", _H_COARSE)
    fld.setNumber(1, "XMin", 0.02e-3)
    fld.setNumber(1, "XMax", 0.24e-3)
    fld.setNumber(1, "YMin", _Y0 - 0.06e-3)
    fld.setNumber(1, "YMax", _Y0 + 0.06e-3)
    fld.setNumber(1, "ZMin", _LZ - 0.04e-3)
    fld.setNumber(1, "ZMax", _LZ)
    fld.setNumber(1, "Thickness", 1.5 * _H_COARSE)
    fld.setAsBackgroundMesh(1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.model.add_physical_group(3, [vol], tag=1)
    gmsh.model.add_physical_group(2, top, tag=4)   # TopFace: laser + evaporation
    gmsh.model.add_physical_group(2, bot, tag=5)   # BottomFace: convection
    occ.synchronize()
    gmsh.model.mesh.generate(3)
    domain, _, ft = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()

    V = fem.functionspace(domain, ("Lagrange", 1))
    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    Tn = fem.Function(V); Tn.x.array[:] = _T0
    Ts = fem.Function(V, name="Temperature"); Ts.x.array[:] = _T0
    w, tr = ufl.TestFunction(V), ufl.TrialFunction(V)

    def liquid_fraction(T):
        return ufl.conditional(
            ufl.lt(T, _T_SOL), 0.0,
            ufl.conditional(ufl.gt(T, _T_LIQ), 1.0, (T - _T_SOL) / (_T_LIQ - _T_SOL)))

    Q_latent = _RHO * _L_F * (liquid_fraction(Ts) - liquid_fraction(Tn)) / _DT
    pref = 0.82 * _PA * _DHLV / ufl.sqrt(2.0 * np.pi * _RV * Ts)
    Q_evap = pref * ufl.exp((_DHLV / (_T_BOIL * _RV)) * (1.0 - (_T_BOIL / Ts)))
    Q_conv = _H_CONV * (Ts - _T_AMB)
    peak = 2.0 * _A * _P / (np.pi * _R_B**2)
    laser = fem.Function(V)

    def set_laser(t):
        x0 = _laser_x(t)
        laser.interpolate(lambda x: (peak * np.exp(
            -2.0 * ((x[0] - x0)**2 + (x[1] - _Y0)**2 + (x[2] - _LZ)**2) / _R_B**2)
        ).astype(PETSc.ScalarType))

    residual = (
        _RHO * _CP * (Ts - Tn) / _DT * w * dx
        + _K * ufl.dot(ufl.grad(Ts), ufl.grad(w)) * dx
        - laser * w * ds(4)
        + Q_evap * w * ds(4)
        + Q_latent * w * dx
        + Q_conv * w * ds(5)
    )
    problem = NonlinearProblem(residual, Ts, bcs=[], J=ufl.derivative(residual, Ts, tr))
    solver = NewtonSolver(domain.comm, problem)
    ksp = solver.krylov_solver
    ksp.setType("cg"); ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-6, atol=1e-8, max_it=1000)
    ksp.setInitialGuessNonzero(True)
    solver.atol = 1e-8; solver.rtol = 1e-6; solver.max_it = 200; solver.report = False

    for n in range(_N_STEPS):
        set_laser(n * _DT)             # start-of-step laser → matches SG
        Ts.x.array[:] = Tn.x.array
        _, converged = solver.solve(Ts)
        if not converged:
            raise RuntimeError(f"FE Newton diverged at step {n}")
        Tn.x.array[:] = Ts.x.array
    return domain, Ts


def _write_fe_xdmf(domain, Ts, path):
    """Write the FE solution (mesh + P1 field) to XDMF — dolfinx handles the
    dof→vertex ordering and tetra connectivity so ``load_xdmf`` reads it back as
    a proper UnstructuredField."""
    from dolfinx import io

    with io.XDMFFile(domain.comm, str(path), "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(Ts)


# ---------------------------------------------------------------------------
# SG (spectral) run — identical physics
# ---------------------------------------------------------------------------

def _run_sg():
    from fast_heat_solv.core.laser import LaserPath, LaserState
    from fast_heat_solv.physics.spectral_helpers import reconstruct_temperature_DCT
    from fast_heat_solv.solvers import build_solver

    class _CVLaser(LaserPath):
        def get_state(self, t, dt):
            return LaserState(x=_X_START + _V * t, y=_Y0, power=_P, is_on=True, v=(_V, 0.0))

    cfg = {
        "simulation": {"backend": "cpu",
                       "duration": _N_STEPS * _DT, "dt": _DT},
        "domain": {"size": [_LX, _LY, _LZ], "mesh": [_SG_NX, _SG_NY, _SG_NZ]},
        "material": dict(_MAT),
        "laser": {"radius": _R_B, "absorptivity": _A, "power_nominal": _P},
        "io": {},
    }
    ctx = SimulationContext.from_dict(cfg)
    ctx.laser_path = _CVLaser()
    solver = build_solver(ctx)
    solver.initialize(ctx)
    for st in range(_N_STEPS):
        solver.step(st * _DT, _DT)
    T_xyz = reconstruct_temperature_DCT(solver.state.a, solver.state)
    return np.ascontiguousarray(T_xyz.transpose(2, 1, 0))


# ---------------------------------------------------------------------------
# The cross-validation test
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.fenics
@pytest.mark.slow
def test_matches_fenics_nonlinear(tmp_path):
    """SG vs an independent nonlinear P1 backward-Euler FE solve (constant props)."""
    pytest.importorskip("dolfinx")
    pytest.importorskip("gmsh")
    pytest.importorskip("ufl")
    pytest.importorskip("petsc4py")

    from fast_heat_solv.io_utils import write_structured_fields
    from fast_heat_solv.io_utils.compute_L2_error import compare

    # FE reference → native dolfinx XDMF (unstructured, with connectivity).
    domain, Ts = _build_and_solve_fe()
    fe_path = tmp_path / "fe.xdmf"
    _write_fe_xdmf(domain, Ts, fe_path)
    T_fe_max = float(Ts.x.array.max())

    # SG field → structured XDMF/H5.
    T_sg = _run_sg()
    assert not np.isnan(T_sg).any() and not np.isinf(T_sg).any()
    x = np.linspace(0.0, _LX, _SG_NX + 1)
    y = np.linspace(0.0, _LY, _SG_NY + 1)
    z = np.linspace(0.0, _LZ, _SG_NZ + 1)
    sg_base = tmp_path / "sg"
    write_structured_fields(sg_base, {"temperature": T_sg}, x, y, z, time=_N_STEPS * _DT)

    assert T_sg.max() > _T_LIQ and T_fe_max > _T_LIQ   # melt + evaporation engaged

    # Dedicated comparison: load both, interpolate the structured SG field at the
    # FE vertices (the library's "HYBRID" fast path), integrate the error on the
    # FE mesh with lumped vertex volumes. FE is the reference (file A).
    res = compare(fe_path, sg_base.with_suffix(".xmf"),
                  attr_a="Temperature", attr_b="temperature", write_error=False)

    assert res["pct_valid"] > 99.0, f"only {res['pct_valid']:.1f}% of points valid"
    # Peaks must agree tightly (the evaporation balance pins the surface T).
    assert abs(float(T_sg.max()) - T_fe_max) < 30.0
    # Cross-method L2 band (computed on the FE mesh via the dedicated unstructured
    # comparison). Verified ≈ 0.88 % on this config; <0.5 % needs the full long-
    # track campaign. Freeze the *band*, not the value.
    assert res["L2_rel"] < 0.012, f"L2_rel={res['L2_rel']:.5f}"
