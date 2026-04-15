#ifndef SLAMCHAIN_HASHVOXEL_RING_GRIDMAP_HPP
#define SLAMCHAIN_HASHVOXEL_RING_GRIDMAP_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace rvoxel {
// 定义Index和Position类型
struct Index {
  int x, y, z;
};

struct Position {
  double x, y, z;
};

struct Size {
  int x, y, z;
};

struct Length {
  double x, y, z;
};

template <typename T>
struct VoxelGrid {
  // 实际储存的数据
  T data;
  // 记录该格子最后一次写入时的世界绝对索引，用于lazy reset
  int wx = std::numeric_limits<int>::max();
  int wy = std::numeric_limits<int>::max();
  int wz = std::numeric_limits<int>::max();
};

template <typename T>
class RingVoxelMap {
 private:
  // 固定的大小（三维独立）
  int size_x_, size_y_, size_z_;
  // 固定的分辨率
  double resolution_;
  double inv_resolution_;

  Position world_pos_;  // Map Bottom Left在世界坐标系的连续位置
  Index world_idx_;     // Map Bottom Left在世界坐标系中的离散位置
  // 存储任意 T 类型
  std::vector<VoxelGrid<T>> data_;

  inline VoxelGrid<T>& at(const Index& idx) {
    return data_[getLinearIndex(idx.x, idx.y, idx.z)];
  }

  inline const VoxelGrid<T>& cat(const Index& idx) const {
    return data_[getLinearIndex(idx.x, idx.y, idx.z)];
  }

 public:
  RingVoxelMap() {}

  inline void setGeometry(int nx, int ny, int nz, double res) {
    if (nx <= 0) nx = 1;
    if (ny <= 0) ny = 1;
    if (nz <= 0) nz = 1;
    size_x_ = nx;
    size_y_ = ny;
    size_z_ = nz;
    resolution_ = res;
    inv_resolution_ = 1.0 / resolution_;
    world_pos_ = {0, 0, 0};
    world_idx_ = {0, 0, 0};
    data_.resize(size_x_ * size_y_ * size_z_);
  }

  inline void setGeometry(double x, double y, double z, double res) {
    int nx = static_cast<int>(x / res);
    int ny = static_cast<int>(y / res);
    int nz = static_cast<int>(z / res);
    setGeometry(nx, ny, nz, res);
  }

  inline Size getSize() const { return {size_x_, size_y_, size_z_}; }

  inline Length getLength() const {
    return {size_x_ * resolution_, size_y_ * resolution_,
            size_z_ * resolution_};
  }

  // 改变Map在世界的位置，不改变底层存储
  inline void setOrigin(const Position& origin) {
    world_pos_ = origin;
    world_idx_ = toWorldIndex(origin);
  }

  // Move the map center to new position
  inline void setOriginCenter(const Position& center) {
    Position origin;
    Length length = getLength();
    origin.x = center.x - length.x / 2.0;
    origin.y = center.y - length.y / 2.0;
    origin.z = center.z - length.z / 2.0;
    setOrigin(origin);
  }

  /// @brief convert index into linear index
  /// @param lx [0, size_x_)
  /// @param ly [0, size_y_)
  /// @param lz [0, size_z_)
  /// @return linear index
  int getLinearIndex(int lx, int ly, int lz) const {
    return lx + size_x_ * ly + size_x_ * size_y_ * lz;
  }

  /// @brief convert liear index into index
  /// @param idx linear index
  /// @return index
  Index getIndexFromLinearIndex(int idx) const {
    Index index;
    index.z = idx / (size_x_ * size_y_);
    int remainder = idx % (size_x_ * size_y_);
    index.y = remainder / size_x_;
    index.x = remainder % size_x_;
    return index;
  }

  /// @brief 返回世界位置对应的体素在Map的索引
  /// @param pos 世界坐标
  /// @return 索引
  inline Index toIndex(const Position& pos) const {
    return {
        static_cast<int>(std::floor((pos.x - world_pos_.x) * inv_resolution_)),
        static_cast<int>(std::floor((pos.y - world_pos_.y) * inv_resolution_)),
        static_cast<int>(std::floor((pos.z - world_pos_.z) * inv_resolution_))};
  }

  inline Index toWorldIndex(const Position& pos) const {
    return {static_cast<int>(std::floor((pos.x) * inv_resolution_)),
            static_cast<int>(std::floor((pos.y) * inv_resolution_)),
            static_cast<int>(std::floor((pos.z) * inv_resolution_))};
  }

  /// @brief 返回体素在世界坐标系下的位置
  /// @param idx 体素索引
  /// @return 世界位置
  inline Position toPosition(const Index& idx) const {
    return {(world_idx_.x + idx.x + 0.5) * resolution_,
            (world_idx_.y + idx.y + 0.5) * resolution_,
            (world_idx_.z + idx.z + 0.5) * resolution_};
  }

  /// @brief 写入Grid
  /// @param idx 相对于map索引
  /// @param v 值
  inline bool setGrid(const Index& idx, const T& v) {
    if (isValid(idx)) {
      at(idx).data = v;
      at(idx).wx = world_idx_.x + idx.x;
      at(idx).wy = world_idx_.y + idx.y;
      at(idx).wz = world_idx_.z + idx.z;
      return true;
    } else {
      return false;
    }
  }

  /// @brief 写入Grid
  /// @param idx 相对于中心的索引，不是世界索引
  /// @param v 值
  inline bool setGrid(const Index& idx, T&& v) {
    if (isValid(idx)) {
      at(idx).data = std::move(v);
      at(idx).wx = world_idx_.x + idx.x;
      at(idx).wy = world_idx_.y + idx.y;
      at(idx).wz = world_idx_.z + idx.z;
      return true;
    } else {
      return false;
    }
  }

  /// @brief 写入grid
  /// @param pos 世界坐标系的位置
  /// @param v 值
  inline bool setGrid(const Position& pos, const T& v) {
    return setGrid(toIndex(pos), v);
  }

  /// @brief 写入grid
  /// @param pos 世界坐标系的位置
  /// @param v 值
  inline bool setGrid(const Position& pos, T&& v) {
    return setGrid(toIndex(pos), std::move(v));
  }

  inline T& getGrid(const Index& idx) { return at(idx).data; }

  inline bool isValid(const Index& idx) const {
    return (idx.x >= 0 && idx.x < size_x_) && (idx.y >= 0 && idx.y < size_y_) &&
           (idx.z >= 0 && idx.z < size_z_);
  }

  // 用于判断当前是否在视野范围内
  inline bool isInside(const Index& idx) const {
    if (!isValid(idx)) {
      return false;
    }
    const auto& g = cat(idx);
    Index relative_idx = {g.wx - world_idx_.x, g.wy - world_idx_.y,
                          g.wz - world_idx_.z};
    auto f = isValid(relative_idx);
    return f;
  }

  void clear() {
    for (auto&& g : data_) {
      g.wx = std::numeric_limits<int>::max();
      g.wy = std::numeric_limits<int>::max();
      g.wz = std::numeric_limits<int>::max();
    }
  }
};

template <typename T>
class RingVoxelMapIterator {
 private:
  const RingVoxelMap<T>& map_;
  Size mapsize_;
  int linear_idx_ = 0;
  int end_idx_;

 public:
  RingVoxelMapIterator(const RingVoxelMapIterator&) = delete;

  RingVoxelMapIterator(const RingVoxelMapIterator&&) = delete;

  RingVoxelMapIterator(const RingVoxelMap<T>& map) : map_(map) {
    mapsize_ = map_.getSize();
    linear_idx_ = 0;
    end_idx_ = mapsize_.x * mapsize_.y * mapsize_.z;
  }

  ~RingVoxelMapIterator() {}

  Index operator*() { return map_.getIndexFromLinearIndex(linear_idx_); }

  int getLinearIndex() { return linear_idx_; }

  bool EOI() { return linear_idx_ == end_idx_; }

  void operator++(int) { linear_idx_++; }
};

template <typename T>
class RingVoxelMapGroundIterator {
 private:
  RingVoxelMap<T>& map_;
  Size mapsize_;
  int thickness_;

 public:
  RingVoxelMapGroundIterator(RingVoxelMap<T>& map) : map_(map) {
    mapsize_ = map_.getSize();
    thickness_ = 2;
  }

  ~RingVoxelMapGroundIterator() {}

  void setGroundThickness(int thickness) { thickness_ = thickness; }

  void traverse(std::function<void(Index, T&)> fn) {
    for (int x = 0; x < mapsize_.x; x++) {
      for (int y = 0; y < mapsize_.y; y++) {
        int countdown = thickness_;
        // from the bottom to the top, the first some are ground grid
        for (int z = 0; z < mapsize_.z; z++) {
          Index idx{x, y, z};
          if (countdown > 0) {
            if (map_.isInside(idx)) {
              T& g = map_.getGrid(idx);
              fn(idx, g);
              countdown--;
            }
          } else {
            break;
          }
        }
      }
    }
  }
};

template <typename T>
class RingVoxelMapObstacleIterator {
 private:
  RingVoxelMap<T>& map_;
  Size mapsize_;
  int ground_thickness_;
  int obstacle_thickness_;

 public:
  RingVoxelMapObstacleIterator(RingVoxelMap<T>& map) : map_(map) {
    mapsize_ = map_.getSize();
    ground_thickness_ = 2;
    obstacle_thickness_ = 1;
  }

  ~RingVoxelMapObstacleIterator() {}

  void setGroundThickness(int thickness) { ground_thickness_ = thickness; }

  void setObstacleThickness(int thickness) { obstacle_thickness_ = thickness; }

  void traverse(std::function<void(Index, T&)> fn) {
    for (int x = 0; x < mapsize_.x; x++) {
      for (int y = 0; y < mapsize_.y; y++) {
        int ground_countdown = ground_thickness_;
        int obstacle_countdown = obstacle_thickness_;
        // from the bottom to the top, the first some are ground grid
        for (int z = 0; z < mapsize_.z; z++) {
          Index idx{x, y, z};
          if (ground_countdown > 0) {
            if (map_.isInside(idx)) {
              ground_countdown--;
            }
          } else {
            if (obstacle_countdown > 0) {
              if (map_.isInside(idx)) {
                T& g = map_.getGrid(idx);
                fn(idx, g);
                obstacle_countdown--;
              }
            } else {
              break;
            }
          }
        }
      }
    }
  }
};

}  // namespace rvoxel

#endif