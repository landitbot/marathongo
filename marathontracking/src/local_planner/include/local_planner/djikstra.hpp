#ifndef SLAMCHAIN_EGOPLUS_DIJKSTRA_HPP
#define SLAMCHAIN_EGOPLUS_DIJKSTRA_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace slamchain {
/**
 * @brief 高性能 Dijkstra 实现
 * 特性：
 * - 二叉堆优先队列（std::priority_queue）
 * - 线性化内存布局（缓存友好）
 * - 支持26连通体素地图
 * - 内存池复用，避免动态分配
 * - 增量式重置，支持重复查询
 */
class Dijkstra {
 public:
  // 3D 坐标结构
  struct Coord {
    int x, y, z;

    bool operator==(const Coord& other) const {
      return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const Coord& other) const { return !(*this == other); }
  };

  // 节点状态
  struct Node {
    float dist;         // 从起点到当前节点的距离（g值）
    float cost;         // Inflation cost
    float f_score;      // f = g + h（A*使用）
    uint32_t prev_idx;  // 前驱节点索引（0xFFFFFFFF 表示无前驱）
    bool visited;       // 是否已确定最短路径

    Node()
        : dist(std::numeric_limits<float>::infinity()),
          cost(0),
          f_score(std::numeric_limits<float>::infinity()),
          prev_idx(0xFFFFFFFF),
          visited(false) {}
  };

  // 优先队列元素
  struct QueueItem {
    float f_score;  // A*使用f值，Dijkstra使用dist
    uint32_t idx;   // 线性化索引

    bool operator>(const QueueItem& other) const {
      return f_score > other.f_score;
    }
  };

  // 路径结果
  struct PathResult {
    std::vector<Coord> path;  // 路径点序列（包含起点和终点）
    float total_cost;         // 总代价
    bool found;               // 是否找到路径
    int nodes_expanded;       // 扩展的节点数（用于性能分析）

    PathResult()
        : total_cost(std::numeric_limits<float>::infinity()),
          found(false),
          nodes_expanded(0) {}
  };

  // 启发函数类型
  enum class HeuristicType {
    NONE,       // Dijkstra（h=0）
    EUCLIDEAN,  // 欧氏距离 √((dx)²+(dy)²+(dz)²)
    MANHATTAN,  // 曼哈顿距离 |dx|+|dy|+|dz|
    OCTILE,     // 8方向网格专用（2D）/ 26方向近似
    DIAGONAL    // 考虑对角线移动的精确启发
  };

  // 邻居偏移（26连通）
  static constexpr int NEIGHBOR_COUNT = 26;
  static constexpr int DX[NEIGHBOR_COUNT] = {-1, -1, -1, 0, 0, 0, 1, 1, 1,
                                             -1, -1, -1, 0, 0, 1, 1, 1, -1,
                                             -1, -1, 0,  0, 0, 1, 1, 1};
  static constexpr int DY[NEIGHBOR_COUNT] = {-1, 0, 1,  -1, 0, 1,  -1, 0, 1,
                                             -1, 0, 1,  -1, 1, -1, 0,  1, -1,
                                             0,  1, -1, 0,  1, -1, 0,  1};
  static constexpr int DZ[NEIGHBOR_COUNT] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                                             0,  0,  0,  0,  0,  0,  0,  0,  1,
                                             1,  1,  1,  1,  1,  1,  1,  1};
  // 预计算的边权重（轴对齐=1，面对角线=√2，体对角线=√3）
  static constexpr float EDGE_WEIGHT[NEIGHBOR_COUNT] = {
      1.732f, 1.414f, 1.732f, 1.414f, 1.0f,   1.414f, 1.732f, 1.414f, 1.732f,
      1.414f, 1.0f,   1.414f, 1.0f,   1.0f,   1.414f, 1.0f,   1.414f, 1.732f,
      1.414f, 1.732f, 1.414f, 1.0f,   1.414f, 1.732f, 1.414f, 1.732f};

 public:
  /**
   * @brief 构造函数
   * @param width  地图宽度（x）
   * @param height 地图高度（y）
   * @param depth  地图深度（z）
   */
  Dijkstra(int width, int height, int depth);

  ~Dijkstra();

  // 禁止拷贝（避免内存问题）
  Dijkstra(const Dijkstra&) = delete;
  Dijkstra& operator=(const Dijkstra&) = delete;

  /**
   * @brief 设置障碍物地图
   * @param occupancy 一维占用地图（true=占用），大小必须为 width*height*depth
   *                布局：idx = (z * height + y) * width + x
   */
  void setObstacleMap(const std::vector<bool>& occupancy);

  /**
   * @brief 设置单个障碍物（动态更新）
   */
  void setObstacle(int x, int y, int z, bool occupied);

  void setCost(int x, int y, int z, float cost);

  /**
   * @brief 检查坐标是否可通行
   */
  bool isValid(int x, int y, int z) const;

  /**
   * @brief 检查坐标是否障碍
   */
  bool isObstacle(int x, int y, int z) const;

  /**
   * @brief 执行 Dijkstra 搜索（单源全目标）
   * @param start 起点坐标
   * @return 是否成功初始化
   */
  bool search(const Coord& start);

  /**
   * @brief 执行 A* 搜索（单源单目标，更快）
   * @param start 起点坐标
   * @param goal  目标坐标
   * @param heuristic 启发函数类型（默认欧氏距离）
   * @return 是否找到路径
   */
  bool searchAStar(const Coord& start, const Coord& goal,
                   HeuristicType heuristic = HeuristicType::EUCLIDEAN);

  /**
   * @brief 获取到特定目标的路径
   * @param goal 目标坐标
   * @return 路径结果
   */
  PathResult getPath(const Coord& goal) const;

  /**
   * @brief 快速查询：Dijkstra搜索+返回路径（一次性）
   */
  PathResult query(const Coord& start, const Coord& goal);

  /**
   * @brief 快速查询：A*搜索+返回路径（一次性，推荐）
   */
  PathResult queryAStar(const Coord& start, const Coord& goal,
                        HeuristicType heuristic = HeuristicType::EUCLIDEAN);

  /**
   * @brief 获取某点的最短距离（用于代价图）
   */
  float getDistance(int x, int y, int z) const;

  /**
   * @brief 重置算法状态（保留地图，清除距离）
   */
  void reset();

  /**
   * @brief 获取地图尺寸
   */
  int getWidth() const { return width_; }
  int getHeight() const { return height_; }
  int getDepth() const { return depth_; }
  int getTotalVoxels() const { return total_voxels_; }

 private:
  // 线性化索引：idx = (z * height + y) * width + x
  inline uint32_t coordToIdx(int x, int y, int z) const {
    // return static_cast<uint32_t>((z * height_ + y) * width_ + x);
    return static_cast<uint32_t>(x + width_ * y + width_ * height_ * z);
  }

  inline Coord idxToCoord(uint32_t idx) const {
    // int x = idx % width_;
    // int tmp = idx / width_;
    // int y = tmp % height_;
    // int z = tmp / height_;
    int z = idx / (width_ * height_);
    int remainder = idx % (width_ * height_);
    int y = remainder / width_;
    int x = remainder % width_;
    return {x, y, z};
  }

  // 启发函数计算
  inline float computeHeuristic(const Coord& a, const Coord& b,
                                HeuristicType type) const {
    int dx = std::abs(a.x - b.x);
    int dy = std::abs(a.y - b.y);
    int dz = std::abs(a.z - b.z);

    switch (type) {
      case HeuristicType::NONE:
        return 0.0f;

      case HeuristicType::MANHATTAN:
        return static_cast<float>(dx + dy + dz);

      case HeuristicType::EUCLIDEAN:
        return std::sqrt(static_cast<float>(dx * dx + dy * dy + dz * dz));

      case HeuristicType::DIAGONAL: {
        // 考虑26连通的精确下界
        int min_v = std::min({dx, dy, dz});
        int max_v = std::max({dx, dy, dz});
        int mid_v = dx + dy + dz - min_v - max_v;

        // 尽可能走对角线，然后走面对角线，最后走直线
        return 1.732f * min_v + 1.414f * (mid_v - min_v) +
               1.0f * (max_v - mid_v);
      }

      case HeuristicType::OCTILE:
      default:
        // 简化的对角线启发（较快）
        return std::sqrt(static_cast<float>(dx * dx + dy * dy + dz * dz));
    }
  }

  // 重置节点状态（快速清零）
  void resetNodes();

 private:
  // 地图尺寸
  const int width_;
  const int height_;
  const int depth_;
  const int total_voxels_;

  // 数据存储
  std::vector<bool> obstacle_map_;  // 障碍物地图
  std::vector<Node> nodes_;         // 节点状态（内存池）

  // 优先队列（使用 std::priority_queue，二叉堆实现）
  using PriorityQueue = std::priority_queue<QueueItem, std::vector<QueueItem>,
                                            std::greater<QueueItem>>;

  // 起点缓存（用于增量更新判断）
  Coord last_start_;
  Coord last_goal_;  // A*缓存目标
  bool has_start_;
  bool has_goal_;
  HeuristicType last_heuristic_;
};

inline Dijkstra::Dijkstra(int width, int height, int depth)
    : width_(width),
      height_(height),
      depth_(depth),
      total_voxels_(width * height * depth),
      has_start_(false),
      has_goal_(false),
      last_heuristic_(HeuristicType::NONE) {
  if (width <= 0 || height <= 0 || depth <= 0) {
    throw std::invalid_argument("Map dimensions must be positive");
  }

  // 预分配内存
  obstacle_map_.resize(total_voxels_, false);  // 默认全空闲
  nodes_.resize(total_voxels_);

  last_start_ = {-1, -1, -1};
  last_goal_ = {-1, -1, -1};
}

inline Dijkstra::~Dijkstra() = default;

inline void Dijkstra::setObstacleMap(const std::vector<bool>& occupancy) {
  if (static_cast<int>(occupancy.size()) != total_voxels_) {
    throw std::invalid_argument("Occupancy map size mismatch");
  }
  obstacle_map_ = occupancy;
  has_start_ = false;  // 地图变化，需要重新搜索
  has_goal_ = false;
}

inline void Dijkstra::setObstacle(int x, int y, int z, bool occupied) {
  if (x < 0 || x >= width_ || y < 0 || y >= height_ || z < 0 || z >= depth_) {
    return;  // 越界忽略
  }
  obstacle_map_[coordToIdx(x, y, z)] = occupied;
  has_start_ = false;  // 地图变化，需要重新搜索
  has_goal_ = false;
}

inline void Dijkstra::setCost(int x, int y, int z, float cost) {
  if (x < 0 || x >= width_ || y < 0 || y >= height_ || z < 0 || z >= depth_) {
    return;  // 越界忽略
  }
  nodes_[coordToIdx(x, y, z)].cost = cost;
  has_start_ = false;  // 地图变化，需要重新搜索
  has_goal_ = false;
}

inline bool Dijkstra::isObstacle(int x, int y, int z) const {
  if (x < 0 || x >= width_ || y < 0 || y >= height_ || z < 0 || z >= depth_) {
    return false;  // 越界忽略
  }
  return obstacle_map_[coordToIdx(x, y, z)];
}

inline bool Dijkstra::isValid(int x, int y, int z) const {
  if (x < 0 || x >= width_ || y < 0 || y >= height_ || z < 0 || z >= depth_) {
    return false;
  }
  uint32_t idx = coordToIdx(x, y, z);
  return idx < obstacle_map_.size() && !obstacle_map_[idx];
}

inline void Dijkstra::resetNodes() {
  // 快速重置：使用赋值而非循环（编译器优化为 memset）
  std::fill(nodes_.begin(), nodes_.end(), Node());
}

inline void Dijkstra::reset() {
  resetNodes();
  has_start_ = false;
  has_goal_ = false;
}

/// @brief Dijkstra search
/// @param start start point
/// @return success
inline bool Dijkstra::search(const Coord& start) {
  // 检查起点有效性
  if (!isValid(start.x, start.y, start.z)) {
    return false;
  }

  // 如果起点没变且已有结果，跳过
  if (has_start_ && last_start_ == start && !has_goal_) {
    return true;
  }

  // 重置节点状态
  // resetNodes();

  uint32_t start_idx = coordToIdx(start.x, start.y, start.z);
  nodes_[start_idx].dist = 0.0f;
  nodes_[start_idx].f_score = 0.0f;

  // 优先队列（自动二叉堆）
  PriorityQueue pq;
  pq.push({0.0f, start_idx});

  // 主循环
  while (!pq.empty()) {
    QueueItem curr = pq.top();
    pq.pop();

    Node& curr_node = nodes_[curr.idx];

    // 跳过过时条目（已有更短路径）
    if (curr.f_score > curr_node.f_score) continue;

    // 已访问过则跳过
    if (curr_node.visited) continue;
    curr_node.visited = true;

    Coord curr_coord = idxToCoord(curr.idx);

    // 遍历26邻居
    for (int i = 0; i < NEIGHBOR_COUNT; ++i) {
      int nx = curr_coord.x + DX[i];
      int ny = curr_coord.y + DY[i];
      int nz = curr_coord.z + DZ[i];

      if (!isValid(nx, ny, nz)) continue;

      uint32_t nidx = coordToIdx(nx, ny, nz);
      Node& neighbor = nodes_[nidx];

      if (neighbor.visited) continue;

      float new_dist = curr_node.dist + neighbor.cost + EDGE_WEIGHT[i];

      if (new_dist < neighbor.dist) {
        neighbor.dist = new_dist;
        neighbor.f_score = new_dist;  // Dijkstra: f = g
        neighbor.prev_idx = curr.idx;
        pq.push({new_dist, nidx});
      }
    }
  }

  last_start_ = start;
  has_start_ = true;
  has_goal_ = false;
  return true;
}

/// @brief Astar search
/// @param start start point
/// @param goal end point
/// @param heuristic heuristic type
/// @return
inline bool Dijkstra::searchAStar(const Coord& start, const Coord& goal,
                                  HeuristicType heuristic) {
  // 检查起点和目标有效性
  if (!isValid(start.x, start.y, start.z) || !isValid(goal.x, goal.y, goal.z)) {
    return false;
  }

  // 如果参数没变且已有结果，跳过
  if (has_start_ && has_goal_ && last_start_ == start && last_goal_ == goal &&
      last_heuristic_ == heuristic) {
    return nodes_[coordToIdx(goal.x, goal.y, goal.z)].dist <
           std::numeric_limits<float>::infinity();
  }

  // 重置节点状态
  // resetNodes();

  uint32_t start_idx = coordToIdx(start.x, start.y, start.z);
  uint32_t goal_idx = coordToIdx(goal.x, goal.y, goal.z);

  float initial_h = computeHeuristic(start, goal, heuristic);
  nodes_[start_idx].dist = 0.0f;
  nodes_[start_idx].f_score = initial_h;

  // 优先队列（按 f = g + h 排序）
  PriorityQueue pq;
  pq.push({initial_h, start_idx});

  // 主循环
  while (!pq.empty()) {
    QueueItem curr = pq.top();
    pq.pop();

    Node& curr_node = nodes_[curr.idx];

    // 跳过过时条目
    if (curr.f_score > curr_node.f_score) continue;

    // 已访问过则跳过
    if (curr_node.visited) continue;
    curr_node.visited = true;

    // 到达目标，提前终止（A*的关键优化）
    if (curr.idx == goal_idx) {
      break;
    }

    Coord curr_coord = idxToCoord(curr.idx);

    // 遍历26邻居
    for (int i = 0; i < NEIGHBOR_COUNT; ++i) {
      int nx = curr_coord.x + DX[i];
      int ny = curr_coord.y + DY[i];
      int nz = curr_coord.z + DZ[i];

      if (!isValid(nx, ny, nz)) continue;

      uint32_t nidx = coordToIdx(nx, ny, nz);
      Node& neighbor = nodes_[nidx];

      if (neighbor.visited) continue;

      float new_dist = curr_node.dist + neighbor.cost + EDGE_WEIGHT[i];

      if (new_dist < neighbor.dist) {
        neighbor.dist = new_dist;
        neighbor.prev_idx = curr.idx;

        // 计算启发值
        Coord neighbor_coord = {nx, ny, nz};
        float h = computeHeuristic(neighbor_coord, goal, heuristic);
        neighbor.f_score = new_dist + h;

        pq.push({neighbor.f_score, nidx});
      }
    }
  }

  last_start_ = start;
  last_goal_ = goal;
  has_start_ = true;
  has_goal_ = true;
  last_heuristic_ = heuristic;

  // 返回是否找到路径
  return nodes_[goal_idx].dist < std::numeric_limits<float>::infinity();
}

inline Dijkstra::PathResult Dijkstra::getPath(const Coord& goal) const {
  PathResult result;

  if (!has_start_ || !isValid(goal.x, goal.y, goal.z)) {
    return result;
  }

  uint32_t goal_idx = coordToIdx(goal.x, goal.y, goal.z);
  const Node& goal_node = nodes_[goal_idx];

  if (goal_node.dist == std::numeric_limits<float>::infinity()) {
    return result;  // 不可达
  }

  result.found = true;
  result.total_cost = goal_node.dist;

  // 回溯路径
  std::vector<Coord> reverse_path;
  uint32_t curr_idx = goal_idx;
  const uint32_t INVALID_IDX = 0xFFFFFFFF;
  int node_count = 0;

  while (curr_idx != INVALID_IDX && node_count < total_voxels_) {
    reverse_path.push_back(idxToCoord(curr_idx));
    if (curr_idx >= nodes_.size()) break;  // 安全检查
    curr_idx = nodes_[curr_idx].prev_idx;
    node_count++;
  }

  // 反转得到起点到终点的顺序
  result.path.assign(reverse_path.rbegin(), reverse_path.rend());
  result.nodes_expanded = node_count;
  return result;
}

inline Dijkstra::PathResult Dijkstra::query(const Coord& start,
                                            const Coord& goal) {
  if (!search(start)) {
    return PathResult();
  }
  return getPath(goal);
}

inline Dijkstra::PathResult Dijkstra::queryAStar(const Coord& start,
                                                 const Coord& goal,
                                                 HeuristicType heuristic) {
  PathResult result;

  if (!searchAStar(start, goal, heuristic)) {
    return result;
  }

  result = getPath(goal);

  // 统计扩展节点数（用于性能分析）
  int expanded = 0;
  for (const auto& node : nodes_) {
    if (node.visited) expanded++;
  }
  result.nodes_expanded = expanded;

  return result;
}

inline float Dijkstra::getDistance(int x, int y, int z) const {
  if (!isValid(x, y, z)) {
    return std::numeric_limits<float>::infinity();
  }
  return nodes_[coordToIdx(x, y, z)].dist;
}

}  // namespace slamchain

#endif  // SLAMCHAIN_EGOPLUS_DIJKSTRA_HPP