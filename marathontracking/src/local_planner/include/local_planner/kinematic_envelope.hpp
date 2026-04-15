#ifndef KINEMATIC_ENVELOPE_HPP
#define KINEMATIC_ENVELOPE_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * Conditional Asymmetric Priority-based Multi-Axis Projector
 * Author: Felix
 *
 * 特性：
 * - 支持N维度
 * - 支持自定义包络约束
 * - 支持线性/非线性包络约束
 * - 支持条件包络约束
 * - 无需求导
 * - 支持保护高优先级量
 * - 快速Clip
 * - 离散采样搜索
 */

namespace KE {

struct Dim {
  double min;                    // 最小值
  double max;                    // 最大值
  double adjust_min;             // 在clip中，允许调整到的最小值
  double adjust_max;             // 在clip中，允许调整到的最大值
  unsigned int adjust_priority;  // 安全优先级，越大则尽可能减少调整这个值
  std::vector<double> values;    // 离散采样值

  double resolution;

  inline void setAdjustRange(double min, double max) {
    adjust_min = min;
    adjust_max = max;
  }

  Dim(double min_val, double max_val, int count, unsigned int priority)
      : min(min_val), max(max_val), adjust_priority(priority) {
    adjust_min = min_val;
    adjust_max = max_val;

    // 保证数值安全性
    {
      if (count < 5) {
        count = 10;
      }
      resolution = (max - min) / (count - 1);
      if (resolution > 0.05) {
        count = (max_val - min_val) / 0.05 + 1;
        resolution = (max - min) / (count - 1);
      }
    }

    values.resize(count);
    for (int i = 0; i < count; ++i) {
      values[i] = min + i * resolution;
    }
  }
};

struct ClipResult {
  bool success;                // 是否成功
  bool clipped;                // 是否真的发生了裁剪
  std::vector<double> values;  // 裁剪后的值
  std::vector<double> deltas;  // 各维度的变化量(原始值 - 裁剪值)

  friend std::ostream& operator<<(std::ostream& out, const ClipResult& r) {
    if (r.success) {
      out << "Success: True" << std::endl;
    } else {
      out << "Success: False" << std::endl;
    }

    if (r.clipped) {
      out << "Clipped: True" << std::endl;
    } else {
      out << "Clipped: False" << std::endl;
    }

    {
      out << "Values: ";
      for (size_t i = 0; i < r.values.size(); i++) {
        double v = r.values[i];
        if (i == 0) {
          out << "[" << std::setprecision(4) << std::fixed << v << ", ";
        } else if (i == r.values.size() - 1) {
          out << std::setprecision(4) << std::fixed << v << "]";
        } else {
          out << std::setprecision(4) << std::fixed << v << ", ";
        }
      }
      out << std::endl;
    }

    {
      out << "Deltas: ";
      for (size_t i = 0; i < r.deltas.size(); i++) {
        double v = r.deltas[i];
        if (i == 0) {
          out << "[" << std::setprecision(4) << std::fixed << v << ", ";
        } else if (i == r.deltas.size() - 1) {
          out << std::setprecision(4) << std::fixed << v << "]";
        } else {
          out << std::setprecision(4) << std::fixed << v << ", ";
        }
      }
      out << std::endl;
    }
    return out;
  }
};

class EnvelopeBase {
 public:
  EnvelopeBase() = default;

  virtual ~EnvelopeBase() = default;

  void setXName(const std::string& name) { x_name_ = name; }

  void setYName(const std::string& name) { y_name_ = name; }

  const std::string& getXName() { return x_name_; }

  const std::string& getYName() { return y_name_; }

  void setXIndex(int idx) { x_idx = idx; }

  void setYIndex(int idx) { y_idx = idx; }

  inline int getXIndex() { return x_idx; }

  inline int getYIndex() { return y_idx; }

  virtual bool valid(double x, double y) = 0;

  virtual bool workable(double x, double y) {
    if (condition_fn_ == nullptr) {
      return true;
    } else {
      return condition_fn_(x, y);
    }
  }

  void setCondition(std::function<bool(double x, double y)> fn) {
    condition_fn_ = fn;
  }

 private:
  std::string x_name_;
  std::string y_name_;
  int x_idx;
  int y_idx;
  std::function<bool(double x, double y)> condition_fn_ = nullptr;
};

template <char Sym>
class LinearEnvelope : public EnvelopeBase {
 public:
  LinearEnvelope() = default;
  ~LinearEnvelope() = default;

  static std::unique_ptr<LinearEnvelope<Sym>> create() {
    return std::make_unique<LinearEnvelope<Sym>>();
  }

  void config(double kx, double ky, double b, double eps = 0.1) {
    kx_ = kx;
    ky_ = ky;
    b_ = b;
    eps_ = eps;
  }

  void setXAbs(bool v) { abs_x_ = v; }

  void setYAbs(bool v) { abs_y_ = v; }

  bool valid(double x, double y) override {
    double x1 = abs_x_ ? std::abs(x) : x;
    double y1 = abs_y_ ? std::abs(y) : y;

    double b = kx_ * x1 + ky_ * y1;
    if constexpr (Sym == '=') {
      return std::abs(b - b_) < eps_;
    } else if constexpr (Sym == '>') {
      return b >= b_;
    } else if constexpr (Sym == '<') {
      return b <= b_;
    }
  }

 private:
  double kx_ = 0;
  double ky_ = 0;
  double b_ = 0;
  double eps_ = 0.1;
  bool abs_x_ = false;
  bool abs_y_ = false;
};

class KinematicEnvelope {
 private:
  std::unordered_map<std::string, int> dim_index_map_;
  std::vector<Dim> dims_;
  std::vector<std::unique_ptr<EnvelopeBase>> envelopes_;

  std::vector<int> adjust_order_;  // generated by compile()

  std::size_t max_iteration_ = 10;

  bool isConflicted(const std::vector<double>& val) {
    for (auto&& enve : envelopes_) {
      int xidx = enve->getXIndex();
      int yidx = enve->getYIndex();
      double x = val[xidx];
      double y = val[yidx];

      if (!enve->workable(x, y)) {
        continue;
      }

      if (!enve->valid(x, y)) {
        return true;
      }
    }
    return false;
  }

  std::vector<double> getValidSamples(const std::vector<double>& val,
                                      int dim_idx) {
    // get dim info
    auto& dim = dims_[dim_idx];

    // copy
    std::vector<double> test_val = val;

    // used to store the valid values
    std::vector<double> valid_values;
    valid_values.reserve(dim.values.size());

    // 根据adjust_min/adjust_max减少遍历的次数
    int beg_idx = (dim.adjust_min - dim.min) / dim.resolution - 5;
    int end_idx = (dim.adjust_max - dim.min) / dim.resolution + 5;
    beg_idx = std::clamp<int>(beg_idx, 0, dim.values.size() - 1);
    end_idx = std::clamp<int>(end_idx, 0, dim.values.size());

    for (int i = beg_idx; i < end_idx; i++) {
      test_val[dim_idx] = dim.values[i];
      if (isConflicted(test_val)) {
        continue;
      }
      valid_values.push_back(dim.values[i]);
    }
    return valid_values;
  }

  ClipResult clipImple(const std::vector<double>& val) {
    std::cout << "-----------------" << std::endl;
    ClipResult result;
    result.values = val;
    result.deltas.resize(val.size(), 0);
    result.success = false;
    result.clipped = false;

    if (val.size() != dims_.size()) {
      throw std::runtime_error("clipImple: val.size() != dims_.size()");
    }

    // 预处理：基础范围Clip
    for (size_t i = 0; i < dims_.size(); ++i) {
      auto it_start =
          std::lower_bound(dims_[i].values.begin(), dims_[i].values.end(),
                           dims_[i].adjust_min + 1e-4);
      auto it_end =
          std::upper_bound(dims_[i].values.begin(), dims_[i].values.end(),
                           dims_[i].adjust_max - 1e-4);

      if (it_start == dims_[i].values.end() || it_start >= it_end) {
        // 如果范围内没点，找全集里最接近 adjust_min 的一个点保底
        result.values[i] =
            (dims_[i].adjust_min < dims_[i].min)
                ? dims_[i].min
                : (dims_[i].adjust_min > dims_[i].max ? dims_[i].max
                                                      : dims_[i].values[0]);
        continue;
      }

      double safe_min_sample = *it_start;
      double safe_max_sample = *std::prev(it_end);

      double clamped =
          std::clamp(result.values[i], safe_min_sample, safe_max_sample);

      if (std::abs(clamped - result.values[i]) > 1e-7) {
        result.values[i] = clamped;
        result.clipped = true;
      }
    }

    // 迭代次数
    for (int iter_i = 0; iter_i < max_iteration_; iter_i++) {
      if (!isConflicted(result.values)) {
        result.success = true;
        break;
      }

      // 如果有冲突, 则进入按权重调整流程
      for (auto&& adjusting_dim_idx : adjust_order_) {
        // 计算当前状态下有效的采样
        auto valid_samples = getValidSamples(result.values, adjusting_dim_idx);
        if (valid_samples.empty()) {
          // 如果这里无法继续调整了, 则试一试调整一下其他的维度,
          // 等待下一轮调整再继续调整
          continue;
        }

        // 寻找离原值 val[i] 最近的采样点
        auto it =
            std::min_element(valid_samples.begin(), valid_samples.end(),
                             [&](double a, double b) {
                               return std::abs(a - val[adjusting_dim_idx]) <
                                      std::abs(b - val[adjusting_dim_idx]);
                             });

        // 边界检查
        if (*it < dims_[adjusting_dim_idx].adjust_min ||
            *it > dims_[adjusting_dim_idx].adjust_max) {
          // 超出调整范围，放弃本次调整
          continue;
        }

        // 更新值
        double delta = *it - result.values[adjusting_dim_idx];
        if (std::abs(delta) > 1e-7) {
          result.values[adjusting_dim_idx] = *it;
          result.clipped = true;
        }
      }
    }

    if (!result.success) {
      // TODO: 进入极其耗时的遍历整个空间，找到欧氏距离最近的解
    }

    // 计算最终deltas
    std::transform(val.begin(), val.end(), result.values.begin(),
                   result.deltas.begin(), std::minus<double>());

    return result;
  }

 public:
  KinematicEnvelope() {}
  ~KinematicEnvelope() {}

  KinematicEnvelope(const KinematicEnvelope&) = delete;
  KinematicEnvelope& operator=(const KinematicEnvelope&) = delete;

  KinematicEnvelope(KinematicEnvelope&&) = default;
  KinematicEnvelope& operator=(KinematicEnvelope&&) = default;

  int getDimIndex(const std::string& name) {
    auto it = dim_index_map_.find(name);
    if (it == dim_index_map_.end()) {
      return -1;
    }
    return it->second;
  }

  void addDim(const std::string& dim_name, double min, double max,
              std::size_t count, unsigned int priority) {
    dim_index_map_[dim_name] = dim_index_map_.size();
    dims_.emplace_back(min, max, count, priority);
  }

  void addEnvelope(std::unique_ptr<EnvelopeBase>&& envelope) {
    int xi = getDimIndex(envelope->getXName());
    int yi = getDimIndex(envelope->getYName());
    if (xi == -1 || yi == -1) {
      throw std::runtime_error("Dimension name not found in map!");
    }
    envelope->setXIndex(xi);
    envelope->setYIndex(yi);
    envelopes_.push_back(std::move(envelope));
  }

  void compile() {
    // pre-compute order
    {
      adjust_order_.resize(dims_.size());
      std::iota(adjust_order_.begin(), adjust_order_.end(), 0);
      std::sort(adjust_order_.begin(), adjust_order_.end(), [&](int a, int b) {
        return dims_[a].adjust_priority < dims_[b].adjust_priority;
      });
    }
  }

  void setMaxIteration(std::size_t n) { max_iteration_ = n; }

  void setAdjustRange(int dim, double min, double max) {
    dims_[dim].setAdjustRange(min, max);
  }

  void setAdjustRange(const std::vector<double>& min,
                      const std::vector<double>& max) {
    if (min.size() != max.size()) {
      throw std::runtime_error("setAdjustRange(): min.size() != max.size()");
    }

    if (min.size() != dims_.size()) {
      throw std::runtime_error("setAdjustRange(): min.size() != dims_.size()");
    }

    for (size_t i = 0; i < min.size(); i++) {
      dims_[i].setAdjustRange(min[i], max[i]);
    }
  }

  void setAdjustRangeDelta(const std::vector<double>& val,
                           const std::vector<double>& delta) {
    if (val.size() != delta.size()) {
      throw std::runtime_error("setAdjustRange(): val.size() != delta.size()");
    }

    if (val.size() != dims_.size()) {
      throw std::runtime_error("setAdjustRange(): val.size() != dims_.size()");
    }

    for (size_t i = 0; i < val.size(); i++) {
      double _min = std::clamp(val[i] - delta[i], dims_[i].min, dims_[i].max);
      double _max = std::clamp(val[i] + delta[i], dims_[i].min, dims_[i].max);
      dims_[i].setAdjustRange(_min, _max);
    }
  }

  ClipResult clip(const std::vector<double>& val) { return clipImple(val); }
};

}  // namespace KE

#endif  // KINEMATIC_ENVELOPE_HPP