#!/usr/bin/env bash
# Rebuild the eigen_spmv extension.
# BIT-EXACTNESS INVARIANT: these flags must not change. In particular NO
# -march=native / -mavx* / -ffast-math (SIMD width changes Eigen's reduction
# order and breaks bit-exact alignment with OR-Tools PDLP).
# Toolchain of record: g++ 13.3.0 / glibc 2.39 / Eigen 3.4.0 (bundled) / SSE2.
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -std=c++17 -fPIC -fwrapv -DEIGEN_MPL2_ONLY -shared \
  -I../third_party \
  -I../third_party/pybind11_root \
  -I"$(python -c "import sysconfig; print(sysconfig.get_path('include'))")" \
  eigen_spmv.cpp -o "eigen_spmv$(python3-config --extension-suffix)"
echo "built: eigen_spmv$(python3-config --extension-suffix)"
