#!/bin/bash
set -x
mv implementations/factories/* src/fast_heat_solv/factories/ || true
mv implementations/file_io/* src/fast_heat_solv/io_utils/ || true
mv implementations/solvers/* src/fast_heat_solv/solvers/ || true
mv implementations/physics/* src/fast_heat_solv/physics/ || true
mv core/standalone_runner.py src/fast_heat_solv/solver_base.py || true
mv core/* src/fast_heat_solv/core/ || true
mv utils/spectral_helpers.py src/fast_heat_solv/physics/ || true
mv utils/comparison.py tests/ || true
mv utils/eagar_tsai.py tests/ || true
mv utils/loader.py tests/ || true
mv utils/cut_views.py research/ || true
mv utils/gcode_path.py src/fast_heat_solv/io_utils/ || true

# clean up
rm -rf implementations utils core
