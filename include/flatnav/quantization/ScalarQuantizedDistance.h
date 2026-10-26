#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/IPDistanceDispatcher.h>
#include <flatnav/distances/L2DistanceDispatcher.h>
#include <flatnav/util/Datatype.h>
#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_set>
#include <stdexcept>
#include <vector>

namespace flatnav::quantization {

using flatnav::distances::DistanceInterface;
using flatnav::distances::IPDistanceDispatcher;
using flatnav::distances::L2DistanceDispatcher;
using flatnav::distances::MetricType;
using flatnav::util::DataType;

/// Scalar quantizer that maps fp32 vectors to int8 using per-dimension
/// min/max uniform quantization (8-bit). 
template <MetricType metric>
class ScalarQuantizedDistance
    : public DistanceInterface<ScalarQuantizedDistance<metric>> {

  friend class DistanceInterface<ScalarQuantizedDistance<metric>>;

 public:
  ScalarQuantizedDistance() = default;

  explicit ScalarQuantizedDistance(size_t dim)
      : _dimension(dim),
        _min_vals(dim, std::numeric_limits<float>::max()),
        _max_vals(dim, std::numeric_limits<float>::lowest()),
        _scales(dim, 0.0f),
        _is_trained(false) {}

  static std::unique_ptr<ScalarQuantizedDistance<metric>> create(size_t dim) {
    return std::make_unique<ScalarQuantizedDistance<metric>>(dim);
  }

  /// Learn per-dimension min/max from training data and derive scales.
  /// @param data               Pointer to row-major fp32 training vectors.
  /// @param n                  Number of training vectors.
  /// @param max_train_samples  If > 0 and < n, sample this many rows for
  ///                           min/max computation instead of scanning all n.
  void train(const float* data, size_t n, size_t max_train_samples = 0) {
    if (n == 0) {
      throw std::invalid_argument(
          "ScalarQuantizedDistance::train: need at least one training vector.");
    }

    std::fill(_min_vals.begin(), _min_vals.end(),
              std::numeric_limits<float>::max());
    std::fill(_max_vals.begin(), _max_vals.end(),
              std::numeric_limits<float>::lowest());

    if (max_train_samples > 0 && max_train_samples < n) {
      // Generate max_train_samples unique random indices (O(max_train_samples) memory)
      std::unordered_set<size_t> idx_set;
      idx_set.reserve(max_train_samples);
      std::mt19937 rng(42);
      std::uniform_int_distribution<size_t> dist(0, n - 1);
      while (idx_set.size() < max_train_samples) {
        idx_set.insert(dist(rng));
      }
      // Compute per-dimension min and max over sampled rows
      for (size_t idx : idx_set) {
        const float* vec = data + idx * _dimension;
        for (size_t d = 0; d < _dimension; d++) {
          if (vec[d] < _min_vals[d]) _min_vals[d] = vec[d];
          if (vec[d] > _max_vals[d]) _max_vals[d] = vec[d];
        }
      }
    } else {
      // Compute per-dimension min and max over all rows
      for (size_t i = 0; i < n; i++) {
        const float* vec = data + i * _dimension;
        for (size_t d = 0; d < _dimension; d++) {
          if (vec[d] < _min_vals[d]) _min_vals[d] = vec[d];
          if (vec[d] > _max_vals[d]) _max_vals[d] = vec[d];
        }
      }
    }

    // Derive scales: scale[d] = 255.0 / (max[d] - min[d])
    for (size_t d = 0; d < _dimension; d++) {
      float range = _max_vals[d] - _min_vals[d];
      if (range < 1e-10f) {
        // avoid division by zero
        _scales[d] = 0.0f;
      } else {
        _scales[d] = 255.0f / range;
      }
    }

    _is_trained = true;
  }

  bool isTrained() const { return _is_trained; }

  /// Quantize a single fp32 vector to int8.
  /// @param src   Pointer to fp32 input vector of length _dimension.
  /// @param dst   Pointer to int8 output buffer of length _dimension.
  void quantize(const float* src, int8_t* dst) const {
    for (size_t d = 0; d < _dimension; d++) {
      float val = (src[d] - _min_vals[d]) * _scales[d];
      // Clamp to [0, 255], then shift to signed int8 range [-128, 127]
      val = std::max(0.0f, std::min(255.0f, std::round(val)));
      dst[d] = static_cast<int8_t>(static_cast<int>(val) - 128);
    }
  }

  /// Dequantize a single int8 vector back to fp32.
  /// @param src   Pointer to int8 input vector of length _dimension.
  /// @param dst   Pointer to fp32 output buffer of length _dimension.
  void dequantize(const int8_t* src, float* dst) const {
    for (size_t d = 0; d < _dimension; d++) {
      if (_scales[d] == 0.0f) {
        dst[d] = _min_vals[d];
      } else {
        dst[d] =
            (static_cast<float>(src[d]) + 128.0f) / _scales[d] + _min_vals[d];
      }
    }
  }

  size_t dimension() const { return _dimension; }

  // Accessors for testing
  const std::vector<float>& minVals() const { return _min_vals; }
  const std::vector<float>& maxVals() const { return _max_vals; }
  const std::vector<float>& scales() const { return _scales; }

 private:
  size_t _dimension = 0;
  std::vector<float> _min_vals;
  std::vector<float> _max_vals;
  std::vector<float> _scales;
  bool _is_trained = false;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(_dimension, _min_vals, _max_vals, _scales, _is_trained);
  }

  inline size_t getDimension() const { return _dimension; }

  /// Returns the stored data size per vector: dim * sizeof(int8_t).
  inline size_t dataSizeImpl() { return _dimension * sizeof(int8_t); }

  /// Input data type is float32 -- this controls how addBatch casts
  /// user data pointers. If we were to use DataType::int8, we would be 
  /// misleading the Index method that adds nodes.
  inline DataType getDataTypeImpl() const { return DataType::float32; }

  /// Quantize fp32 input to int8 for storage.
  inline void transformDataImpl(void* destination, const void* src) {
    /// NOTE: There is no need to do std::memcpy since we are quantizing and writing directly 
    /// to the memory block that was preallocated during index construction. 
    quantize(static_cast<const float*>(src),
             static_cast<int8_t*>(destination));
  }

  float decoded(const int8_t* code, size_t d) const {
    if (_scales[d] == 0.0f) {
      return _min_vals[d];
    }
    return (static_cast<float>(code[d]) + 128.0f) / _scales[d] + _min_vals[d];
  }

  /// Distances are computed in the original space: stored codes are decoded per
  /// dimension and a float query is used as it is. On the raw codes every
  /// dimension would be weighted by its own scale.
  float distanceImpl(const void* x, const void* y, bool asymmetric = false) const {
    const int8_t* cy = static_cast<const int8_t*>(y);
    float acc = 0.0f;
    if (asymmetric) {
      const float* q = static_cast<const float*>(x);
      for (size_t d = 0; d < _dimension; d++) {
        float v = decoded(cy, d);
        if constexpr (metric == MetricType::L2) {
          float t = q[d] - v;
          acc += t * t;
        } else {
          acc += q[d] * v;
        }
      }
    } else {
      const int8_t* cx = static_cast<const int8_t*>(x);
      for (size_t d = 0; d < _dimension; d++) {
        float a = decoded(cx, d);
        float b = decoded(cy, d);
        if constexpr (metric == MetricType::L2) {
          float t = a - b;
          acc += t * t;
        } else {
          acc += a * b;
        }
      }
    }
    if constexpr (metric == MetricType::L2) {
      return acc;
    }
    return 1.0f - acc;
  }

  void getSummaryImpl() {
    std::cout << "\nScalarQuantizedDistance Parameters" << std::flush;
    std::cout << "\n-----------------------------\n" << std::flush;
    std::cout << "Metric: "
              << (metric == MetricType::L2 ? "L2" : "IP") << "\n"
              << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
    std::cout << "Trained: " << (_is_trained ? "yes" : "no") << "\n"
              << std::flush;
  }
};

}  // namespace flatnav::quantization
