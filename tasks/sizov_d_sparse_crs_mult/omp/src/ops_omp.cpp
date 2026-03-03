#include "sizov_d_sparse_crs_mult/omp/include/ops_omp.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sizov_d_sparse_crs_mult/common/include/common.hpp"

namespace sizov_d_sparse_crs_mult {

SizovDSparseCRSMultOMP::SizovDSparseCRSMultOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SizovDSparseCRSMultOMP::ValidationImpl() {
  const auto &input = GetInput();
  const auto &A = std::get<0>(input);
  const auto &B = std::get<1>(input);

  if (A.cols != B.rows) {
    return false;
  }
  if (A.row_ptr.size() != A.rows + 1) {
    return false;
  }
  if (B.row_ptr.size() != B.rows + 1) {
    return false;
  }
  if (A.values.size() != A.col_indices.size()) {
    return false;
  }
  if (B.values.size() != B.col_indices.size()) {
    return false;
  }
  if (A.row_ptr.back() != A.values.size()) {
    return false;
  }
  if (B.row_ptr.back() != B.values.size()) {
    return false;
  }

  return true;
}

bool SizovDSparseCRSMultOMP::PreProcessingImpl() {
  GetOutput() = CRSMatrix{};
  return true;
}

bool SizovDSparseCRSMultOMP::RunImpl() {
  const auto &input = GetInput();
  const auto &A = std::get<0>(input);
  const auto &B = std::get<1>(input);

  CRSMatrix C;
  C.rows = A.rows;
  C.cols = B.cols;

  std::vector<std::vector<std::pair<std::size_t, double>>> row_entries(C.rows);

#pragma omp parallel for default(none) shared(A, B, row_entries)
  for (int64_t i = 0; i < static_cast<int64_t>(A.rows); ++i) {
    std::unordered_map<std::size_t, double> accumulator;

    for (std::size_t a_idx = A.row_ptr[i]; a_idx < A.row_ptr[i + 1]; ++a_idx) {
      const std::size_t k = A.col_indices[a_idx];
      const double a_val = A.values[a_idx];

      for (std::size_t b_idx = B.row_ptr[k]; b_idx < B.row_ptr[k + 1]; ++b_idx) {
        const std::size_t j = B.col_indices[b_idx];
        accumulator[j] += a_val * B.values[b_idx];
      }
    }

    auto &row = row_entries[static_cast<std::size_t>(i)];
    row.reserve(accumulator.size());
    for (const auto &[col, value] : accumulator) {
      if (value != 0.0) {
        row.emplace_back(col, value);
      }
    }

    std::sort(row.begin(), row.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
  }

  C.row_ptr.resize(C.rows + 1, 0);
  for (std::size_t i = 0; i < C.rows; ++i) {
    C.row_ptr[i + 1] = C.row_ptr[i] + row_entries[i].size();
  }

  C.values.reserve(C.row_ptr.back());
  C.col_indices.reserve(C.row_ptr.back());
  for (std::size_t i = 0; i < C.rows; ++i) {
    for (const auto &[col, value] : row_entries[i]) {
      C.col_indices.push_back(col);
      C.values.push_back(value);
    }
  }

  GetOutput() = std::move(C);
  return true;
}

bool SizovDSparseCRSMultOMP::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_sparse_crs_mult
