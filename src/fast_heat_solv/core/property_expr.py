"""Whitelisted math expressions of ``T`` for temperature-dependent properties.

A property branch (e.g. ``k.solid``) is given in config as either a bare
number (a constant) or a string expression of ``T`` such as
``"9.248 + 0.01571 * T"`` or ``"12.41 * exp(-T / 3000.0)"``. :class:`PropertyExpr`
parses that string once against a small node/name/function whitelist using
Python's ``ast`` module. It never calls ``eval`` or ``exec`` on the expression
itself, so there is no sandbox-escape surface. (A restricted-namespace ``eval``
would be escapable through attribute access on any in-scope object.)

The validated AST is then rendered to three targets, so the temperature-dependent
hot path stays fully fused rather than falling back to a per-property Python
callback:

* :meth:`PropertyExpr.__call__`, a NumPy/CuPy-broadcasting evaluator that works
  on a scalar or a full field, on host or device;
* :meth:`PropertyExpr.to_py_source`, a numba-``njit``-compatible Python source
  snippet using ``math.<fn>`` calls, which numba compiles natively, to build
  the fused CPU kernel;
* :meth:`PropertyExpr.to_c_source`, a CUDA C expression, to build the fused GPU
  ``ElementwiseKernel``.

Both source renderers take the target precision, since the literals they emit
are what fixes the precision of the generated kernel.
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

import ast
import math

import numpy as np

__all__ = ["PropertyExpr"]


def _xp(T):
    """Return the array module backing *T* (CuPy if it is a CuPy array, else NumPy).

    Lets the evaluator run unchanged on host (parse-time scalars / NumPy fields)
    and device (CuPy fields in the GPU solver) without importing CuPy eagerly.
    """
    if type(T).__module__.startswith("cupy"):
        import cupy
        return cupy
    return np

# Function names allowed in a property expression. Values are the NumPy/CuPy
# ufunc name (used by the runtime evaluator); the same keys double as the
# ``math.<name>`` call used in the generated numba source (Python's ``math``
# module uses these exact names) and as the base of the CUDA single-precision
# intrinsic (``<name>f``) used in the generated C source.
_NUMPY_FUNC_NAME = {
    "sqrt": "sqrt", "exp": "exp", "log": "log", "log2": "log2", "log10": "log10",
    "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "arcsin", "acos": "arccos", "atan": "arctan", "atan2": "arctan2",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "asinh": "arcsinh", "acosh": "arccosh", "atanh": "arctanh",
    "fabs": "abs", "ceil": "ceil", "floor": "floor", "trunc": "trunc",
    "log1p": "log1p", "expm1": "expm1",
}
_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)
_BINOP_SYMBOL = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _validate(node):
    """Recursively reject anything but arithmetic on ``T`` and whitelisted calls.

    Raises ``ValueError`` on the first disallowed construct (attribute
    access, subscripts, comparisons, comprehensions, unknown names/functions,
    non-numeric literals, ...).
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numeric literals are allowed, got {node.value!r}.")
        return
    if isinstance(node, ast.Name):
        if node.id != "T" and node.id not in _ALLOWED_CONSTANTS:
            raise ValueError(
                f"Unknown name {node.id!r} in property expression "
                f"(only 'T' and {sorted(_ALLOWED_CONSTANTS)} are allowed)."
            )
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ValueError(f"Operator {type(node.op).__name__} is not allowed.")
        _validate(node.left)
        _validate(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ValueError(f"Unary operator {type(node.op).__name__} is not allowed.")
        _validate(node.operand)
        return
    if isinstance(node, ast.Call):
        if (not isinstance(node.func, ast.Name) or node.func.id not in _NUMPY_FUNC_NAME
                or node.keywords or not node.args):
            name = getattr(node.func, "id", "<expr>")
            raise ValueError(f"Function {name!r} is not allowed in a property expression.")
        for arg in node.args:
            _validate(arg)
        return
    raise ValueError(f"{type(node).__name__} is not allowed in a property expression.")


def _eval(node, T, xp):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return T if node.id == "T" else _ALLOWED_CONSTANTS[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, T, xp)
        right = _eval(node.right, T, xp)
        op = type(node.op)
        if op is ast.Add:
            return left + right
        if op is ast.Sub:
            return left - right
        if op is ast.Mult:
            return left * right
        if op is ast.Div:
            return left / right
        return left ** right  # ast.Pow
    if isinstance(node, ast.UnaryOp):
        val = _eval(node.operand, T, xp)
        return -val if isinstance(node.op, ast.USub) else val
    # ast.Call
    fn = getattr(xp, _NUMPY_FUNC_NAME[node.func.id])
    return fn(*(_eval(a, T, xp) for a in node.args))


# Precision-dependent rendering. Every literal is wrapped in an explicit
# ``np.floatNN()`` so numba does not promote the whole expression to double;
# CUDA C selects both the literal suffix and the intrinsic name from the same
# suffix ('sqrtf' in single precision, 'sqrt' in double).
_PY_CAST = {np.dtype(np.float32): "np.float32", np.dtype(np.float64): "np.float64"}
_C_SUFFIX = {np.dtype(np.float32): "f", np.dtype(np.float64): ""}
_C_TYPE = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}


def _precision(dtype, table):
    """Look *dtype* up in a precision table, with a clear error if unsupported."""
    key = np.dtype(dtype)
    try:
        return table[key]
    except KeyError:
        raise ValueError(
            f"Unsupported property precision {key.name!r}; "
            f"expected one of {sorted(t.name for t in table)}."
        ) from None


def _render_py(node, cast):
    """Render as a numba-``njit``-compatible Python source expression.

    Parameters
    ----------
    node : ast.AST
        A node of the validated expression tree.
    cast : str
        Literal wrapper to emit, ``"np.float32"`` or ``"np.float64"``.
    """
    if isinstance(node, ast.Constant):
        return f"{cast}({node.value!r})"
    if isinstance(node, ast.Name):
        return "T" if node.id == "T" else f"{cast}({_ALLOWED_CONSTANTS[node.id]!r})"
    if isinstance(node, ast.BinOp):
        left, right = _render_py(node.left, cast), _render_py(node.right, cast)
        sym = "**" if isinstance(node.op, ast.Pow) else _BINOP_SYMBOL[type(node.op)]
        return f"({left} {sym} {right})"
    if isinstance(node, ast.UnaryOp):
        val = _render_py(node.operand, cast)
        return f"(-{val})" if isinstance(node.op, ast.USub) else f"(+{val})"
    # ast.Call: numba's `math` module mirrors Python's math module naming.
    args = ", ".join(_render_py(a, cast) for a in node.args)
    return f"math.{node.func.id}({args})"


def _render_c(node, suffix):
    """Render as a CUDA C expression.

    Parameters
    ----------
    node : ast.AST
        A node of the validated expression tree.
    suffix : str
        ``"f"`` for single-precision literals and intrinsics, ``""`` for double.
    """
    if isinstance(node, ast.Constant):
        return f"{float(node.value):.9e}{suffix}"
    if isinstance(node, ast.Name):
        if node.id == "T":
            return "T"
        return f"{_ALLOWED_CONSTANTS[node.id]:.9e}{suffix}"
    if isinstance(node, ast.BinOp):
        left, right = _render_c(node.left, suffix), _render_c(node.right, suffix)
        if isinstance(node.op, ast.Pow):
            return f"pow{suffix}({left}, {right})"
        return f"({left} {_BINOP_SYMBOL[type(node.op)]} {right})"
    if isinstance(node, ast.UnaryOp):
        val = _render_c(node.operand, suffix)
        return f"(-{val})" if isinstance(node.op, ast.USub) else f"(+{val})"
    # ast.Call: CUDA single-precision intrinsics are the libm name + 'f'.
    args = ", ".join(_render_c(a, suffix) for a in node.args)
    return f"{node.func.id}{suffix}({args})"


class PropertyExpr:
    """A validated math expression of ``T`` (or a bare constant).

    ``str(expr)`` must either parse as a plain number (a constant) or as an
    expression using ``T``, ``+ - * / **``, the constants ``pi``/``e``, and
    calls to a fixed whitelist of math functions (``sqrt``, ``exp``, ``log``,
    trig/hyperbolic functions, and so on). Anything else raises ``ValueError``
    at construction time: attribute access, subscripts, comprehensions, and
    unknown names or functions.
    """

    __slots__ = ("_const", "_tree", "source")

    def __init__(self, expr):
        self.source = str(expr).strip()
        try:
            self._const = float(self.source)
            self._tree = None
            return
        except ValueError:
            pass
        try:
            tree = ast.parse(self.source, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"Invalid property expression {self.source!r}: {exc}") from None
        _validate(tree)
        self._const = None
        self._tree = tree

    @property
    def is_constant(self) -> bool:
        return self._const is not None

    @property
    def constant_value(self) -> float:
        if self._const is None:
            raise ValueError(f"{self.source!r} is not a constant expression.")
        return self._const

    def __call__(self, T):
        """Evaluate at temperature(s) *T* (Python scalar, NumPy or CuPy array)."""
        if self._const is not None:
            return T * 0.0 + self._const
        return _eval(self._tree, T, _xp(T))

    def to_py_source(self, dtype=np.float32) -> str:
        """Return numba-``njit``-compatible Python source with ``T`` free.

        Parameters
        ----------
        dtype : data-type, optional
            Precision of the emitted literals, ``numpy.float32`` (default) or
            ``numpy.float64``.
        """
        cast = _precision(dtype, _PY_CAST)
        if self._const is not None:
            return f"{cast}({self._const!r})"
        return _render_py(self._tree, cast)

    def to_c_source(self, dtype=np.float32) -> str:
        """Return a CUDA C expression with ``T`` free.

        Parameters
        ----------
        dtype : data-type, optional
            Precision of the emitted literals and intrinsics,
            ``numpy.float32`` (default) or ``numpy.float64``.
        """
        suffix = _precision(dtype, _C_SUFFIX)
        if self._const is not None:
            return f"{self._const:.9e}{suffix}"
        return _render_c(self._tree, suffix)

    def __repr__(self):
        return f"PropertyExpr({self.source!r})"
