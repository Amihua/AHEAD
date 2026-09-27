/*
 * eigen_spmv.cpp
 *
 * Thin pybind11 wrapper around Eigen's CSR sparse matrix-vector product.
 * Uses the SAME Eigen call as ORT PDLP (Eigen::SparseMatrix<double,RowMajor,int>)
 * so the float64 result is bit-for-bit identical to ORT's internal SpMV.
 *
 * Exposed functions (all numpy float64 in / numpy float64 out):
 *   csr_matvec(crow, col, val, m, n, x)    ->  A @ x        (m-vector)
 *   csr_rmatvec(crow, col, val, m, n, y)   ->  A.T @ y      (n-vector)
 *
 * crow : int32[m+1]  CSR row-pointer array
 * col  : int32[nnz]  CSR column-index array
 * val  : f64[nnz]    CSR value array
 * m, n : int         matrix shape
 * x    : f64[n]      input vector for A @ x
 * y    : f64[m]      input vector for A.T @ y
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/SparseCore>

namespace py = pybind11;

using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
using Vec   = Eigen::VectorXd;

/* ------------------------------------------------------------------ */
/* A @ x  (forward matvec, m-vector output)                           */
/* ------------------------------------------------------------------ */
py::array_t<double> csr_matvec(
    py::array_t<int,    py::array::c_style | py::array::forcecast> crow_arr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> val_arr,
    int m, int n,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_arr)
{
    const int* crow = crow_arr.data();
    const int* col  = col_arr.data();
    const double* val = val_arr.data();
    int nnz = (int)val_arr.size();

    // Map existing CSR memory — zero-copy, same as ORT's internal layout
    Eigen::Map<const SpMat> A(m, n, nnz, crow, col, val);
    Eigen::Map<const Vec>   x(x_arr.data(), n);

    // Eigen RowMajor A*x: same call as ORT's MatrixVectorProduct
    Vec result = A * x;

    py::array_t<double> out(m);
    std::copy(result.data(), result.data() + m, out.mutable_data());
    return out;
}

/* ------------------------------------------------------------------ */
/* A.T @ y  (transpose matvec, n-vector output)                       */
/* ------------------------------------------------------------------ */
py::array_t<double> csr_rmatvec(
    py::array_t<int,    py::array::c_style | py::array::forcecast> crow_arr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> val_arr,
    int m, int n,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_arr)
{
    const int* crow = crow_arr.data();
    const int* col  = col_arr.data();
    const double* val = val_arr.data();
    int nnz = (int)val_arr.size();

    Eigen::Map<const SpMat> A(m, n, nnz, crow, col, val);
    Eigen::Map<const Vec>   y(y_arr.data(), m);

    // A.transpose() * y: same as ORT's TransposedMatrixVectorProduct
    Vec result = A.transpose() * y;

    py::array_t<double> out(n);
    std::copy(result.data(), result.data() + n, out.mutable_data());
    return out;
}

/* ------------------------------------------------------------------ */
/* (scale * A) @ x — matches ORT's dual-step expression               */
/*   temp = y - dual_step_size * TransposedConstraintMatrix^T * v     */
/* where Eigen evaluates the product of the SCALED matrix expression: */
/* each term is (scale*a_ij)*v_j, accumulated per row — different     */
/* rounding from scale*(A@v).                                         */
/* ------------------------------------------------------------------ */
py::array_t<double> csr_matvec_scaled(
    py::array_t<int,    py::array::c_style | py::array::forcecast> crow_arr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> val_arr,
    int m, int n, double scale,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_arr)
{
    const int* crow = crow_arr.data();
    const int* col  = col_arr.data();
    const double* val = val_arr.data();
    int nnz = (int)val_arr.size();

    Eigen::Map<const SpMat> A(m, n, nnz, crow, col, val);
    Eigen::Map<const Vec>   x(x_arr.data(), n);

    // Same lazy scaled-matrix product expression as ORT's dual step.
    Vec result = (scale * A) * x;

    py::array_t<double> out(m);
    std::copy(result.data(), result.data() + m, out.mutable_data());
    return out;
}

/* ------------------------------------------------------------------ */
/* ORT dual-step temp, replicated literally:                          */
/*   temp = y - dual_step_size * TransposedConstraintMatrix           */
/*              .transpose() * extrapolated_primal                    */
/* TransposedConstraintMatrix is A^T stored ColMajor int64 in ORT.    */
/* The CSR arrays of A ARE the ColMajor storage of A^T (outer = rows  */
/* of A, inner = column indices), so we Map them directly.            */
/* ------------------------------------------------------------------ */
using SpMatT64 = Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>;

py::array_t<double> ort_dual_temp(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> crow_arr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> col_arr,
    py::array_t<double,       py::array::c_style | py::array::forcecast> val_arr,
    int m, int n, double dual_step_size,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> v_arr)
{
    const std::int64_t* crow = crow_arr.data();
    const std::int64_t* col  = col_arr.data();
    const double* val = val_arr.data();
    std::int64_t nnz = (std::int64_t)val_arr.size();

    // A^T as ColMajor (n rows, m cols): columns = rows of A.
    Eigen::Map<const SpMatT64> AT(n, m, nnz, crow, col, val);
    Eigen::Map<const Vec> y(y_arr.data(), m);
    Vec v = Eigen::Map<const Vec>(v_arr.data(), n);   // aligned copy (VectorXd in ORT)
    Vec y_v = y;                                        // aligned copy of dual

    // Literal ORT expression from ComputeNextDualSolution:
    Vec temp = y_v - dual_step_size * AT.transpose() * v;

    py::array_t<double> out(m);
    std::copy(temp.data(), temp.data() + m, out.mutable_data());
    return out;
}

/* ------------------------------------------------------------------ */
/* Literal replica of ORT's dual-step temp including matrix           */
/* construction: A ColMajor int64 -> materialized transpose ->        */
/* middleCols block -> y - sigma * block.transpose() * v              */
/* ------------------------------------------------------------------ */
py::array_t<double> ort_dual_temp_v2(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> crow_arr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> col_arr,
    py::array_t<double,       py::array::c_style | py::array::forcecast> val_arr,
    int m, int n, double dual_step_size,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> v_arr)
{
    const std::int64_t* crow = crow_arr.data();
    const std::int64_t* col  = col_arr.data();
    const double* val = val_arr.data();
    std::int64_t nnz = (std::int64_t)val_arr.size();

    // ORT: qp.constraint_matrix is ColMajor int64 (m x n).
    // Our CSR(A) arrays are the ColMajor storage of A^T; map that and
    // materialize A = (A^T)^T like ORT's problem construction, then
    // transposed_constraint_matrix_ = A.transpose() (materialized).
    using SpT = SpMatT64;
    Eigen::Map<const SpT> AT_map(n, m, nnz, crow, col, val);
    SpT A = AT_map.transpose();       // ColMajor A, sorted
    SpT AT = A.transpose();           // ORT: transposed_constraint_matrix_

    Eigen::Map<const Vec> y_map(y_arr.data(), m);
    Vec v = Eigen::Map<const Vec>(v_arr.data(), n);
    Vec y_v = y_map;

    // ORT: shard(AT) = AT.middleCols(start, size); 1 shard = full range
    Vec temp = y_v - dual_step_size *
                         AT.middleCols(0, AT.cols()).transpose() * v;

    py::array_t<double> out(m);
    std::copy(temp.data(), temp.data() + m, out.mutable_data());
    return out;
}

/* ------------------------------------------------------------------ */
/* Literal replica of ORT's preprocessing (single shard):             */
/*   LInfRuizRescaling(ruiz_iters) + optional L2NormRescaling,        */
/*   cumulative scaling vectors, applied ONCE at the end via          */
/*   ScaleMatrix: a_ij *= row[i] * col[j].                            */
/* Norms are computed from the ORIGINAL matrix with on-the-fly        */
/* cumulative scalings, with ORT's exact rounding groupings:          */
/*   LInf: (max_i |a_ij * row[i]|) * |col[j]|                         */
/*   L2  : sqrt(sum_i (a_ij*row[i])^2) * |col[j]|                     */
/* then vec[k] /= sqrt(norm[k]) (skip zero norms).                    */
/* Input CSR(A) arrays are mapped as A^T ColMajor; A ColMajor is      */
/* materialized via Eigen transpose (sorted), matching ORT's storage. */
/* Returns (val_scaled in CSR order, row_scaling, col_scaling).       */
/* ------------------------------------------------------------------ */
py::tuple ort_preprocess(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> crow_arr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> col_arr,
    py::array_t<double,       py::array::c_style | py::array::forcecast> val_arr,
    int m, int n, int ruiz_iters, bool l2_rescaling)
{
    const std::int64_t* crow = crow_arr.data();
    const std::int64_t* colx = col_arr.data();
    const double* val = val_arr.data();
    std::int64_t nnz = (std::int64_t)val_arr.size();

    Eigen::Map<const SpMatT64> AT(n, m, nnz, crow, colx, val);  // A^T ColMajor
    SpMatT64 A = AT.transpose();                                 // A  ColMajor

    Vec row_scaling = Vec::Ones(m);
    Vec col_scaling = Vec::Ones(n);

    auto col_linf = [&](const SpMatT64& M, const Vec& inner_scale,
                        const Vec& outer_scale, Vec& out) {
        // ScaledColLInfNorm: per column: (max |a*inner_scale[row]|) * |outer_scale[col]|
        for (std::int64_t c = 0; c < M.outerSize(); ++c) {
            double mx = 0.0;
            for (SpMatT64::InnerIterator it(M, c); it; ++it)
                mx = std::max(mx, std::abs(it.value() * inner_scale[it.row()]));
            out[c] = mx * std::abs(outer_scale[c]);
        }
    };
    auto col_l2 = [&](const SpMatT64& M, const Vec& inner_scale,
                      const Vec& outer_scale, Vec& out) {
        for (std::int64_t c = 0; c < M.outerSize(); ++c) {
            double ss = 0.0;
            for (SpMatT64::InnerIterator it(M, c); it; ++it) {
                const double t = it.value() * inner_scale[it.row()];
                ss += t * t;   // MathUtil::Square
            }
            out[c] = std::sqrt(ss) * std::abs(outer_scale[c]);
        }
    };
    auto div_sqrt = [&](const Vec& divisor, Vec& vec) {
        for (std::int64_t i = 0; i < vec.size(); ++i)
            if (divisor[i] != 0) vec[i] /= std::sqrt(divisor[i]);
    };

    Vec col_norm(n), row_norm(m);
    for (int it = 0; it < ruiz_iters; ++it) {
        col_linf(A,  row_scaling, col_scaling, col_norm);
        col_linf(AT, col_scaling, row_scaling, row_norm);
        div_sqrt(col_norm, col_scaling);
        div_sqrt(row_norm, row_scaling);
    }
    if (l2_rescaling) {
        col_l2(A,  row_scaling, col_scaling, col_norm);
        col_l2(AT, col_scaling, row_scaling, row_norm);
        div_sqrt(col_norm, col_scaling);
        div_sqrt(row_norm, row_scaling);
    }

    // ScaleMatrix on the CSR arrays (== transposed matrix in ORT):
    // it.valueRef() *= row_param[it.row()] * col_param[it.col()]
    // For the transposed call row_param = col_scaling, col_param = row_scaling.
    py::array_t<double> val_out(nnz);
    double* vo = val_out.mutable_data();
    for (std::int64_t i = 0; i < m; ++i)
        for (std::int64_t k = crow[i]; k < crow[i + 1]; ++k)
            vo[k] = val[k] * (col_scaling[colx[k]] * row_scaling[i]);

    py::array_t<double> row_out(m), col_out(n);
    std::copy(row_scaling.data(), row_scaling.data() + m, row_out.mutable_data());
    std::copy(col_scaling.data(), col_scaling.data() + n, col_out.mutable_data());
    return py::make_tuple(val_out, row_out, col_out);
}

/* ------------------------------------------------------------------ */
/* Dense reductions matching ORT's single-shard Sharder ops           */
/* (with num_threads=1 → num_shards=1, ORT computes whole-vector      */
/*  Eigen .dot() / .squaredNorm(); these are the exact same kernels)  */
/* ------------------------------------------------------------------ */
double vdot(
    py::array_t<double, py::array::c_style | py::array::forcecast> a_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> b_arr)
{
    // Copy into aligned VectorXd first: ORT's operands are VectorXd
    // (aligned allocation), and Eigen's redux traversal can depend on
    // operand alignment.
    Vec a = Eigen::Map<const Vec>(a_arr.data(), a_arr.size());
    Vec b = Eigen::Map<const Vec>(b_arr.data(), b_arr.size());
    return a.dot(b);
}

double squared_norm(
    py::array_t<double, py::array::c_style | py::array::forcecast> a_arr)
{
    Vec a = Eigen::Map<const Vec>(a_arr.data(), a_arr.size());
    return a.squaredNorm();
}

PYBIND11_MODULE(eigen_spmv, m) {
    m.doc() = "Eigen CSR SpMV matching ORT PDLP's internal computation";
    m.def("csr_matvec",  &csr_matvec,
          "A @ x  (Eigen RowMajor CSR, same as ORT MatrixVectorProduct)");
    m.def("csr_rmatvec", &csr_rmatvec,
          "A.T @ y  (Eigen RowMajor CSR, same as ORT TransposedMatrixVectorProduct)");
    m.def("csr_matvec_scaled", &csr_matvec_scaled,
          "(scale*A) @ x  (Eigen lazy scaled product, same as ORT dual-step expr)");
    m.def("ort_dual_temp", &ort_dual_temp,
          "y - sigma * A^T(ColMajor).transpose() @ v  (literal ORT dual-step temp)");
    m.def("ort_dual_temp_v2", &ort_dual_temp_v2,
          "same, with ORT-style materialized transpose + middleCols block");
    m.def("ort_preprocess", &ort_preprocess,
          "Literal ORT LInfRuiz + L2 rescaling; returns (val_scaled, row_scaling, col_scaling)");
    m.def("vdot", &vdot,
          "a.dot(b)  (Eigen dense dot, same kernel as ORT single-shard Dot)");
    m.def("squared_norm", &squared_norm,
          "a.squaredNorm()  (Eigen, same kernel as ORT single-shard SquaredNorm)");
}
