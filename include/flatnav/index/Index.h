#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/Macros.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/Reordering.h>
#include <flatnav/util/VisitedSetPool.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using flatnav::distances::DistanceInterface;
using flatnav::util::DataType;
using flatnav::util::VisitedSet;
using flatnav::util::VisitedSetPool;

namespace flatnav {

// dist_t: A distance function implementing DistanceInterface.
// label_t: A fixed-width data type for the label (meta-data) of each point.
template <typename dist_t, typename label_t>
class Index {
  typedef std::pair<float, label_t> dist_label_t;
  // internal node numbering scheme. We might need to change this to uint64_t
  typedef uint32_t node_id_t;
  typedef std::pair<float, node_id_t> dist_node_t;

  // NOTE: by default this is a max-heap. We could make this a min-heap
  // by using std::greater, but we want to use the queue as both a max-heap and
  // min-heap depending on the context.

  struct CompareByFirst {
    constexpr bool operator()(dist_node_t const& a, dist_node_t const& b) const noexcept {
      return a.first < b.first;
    }
  };

  typedef std::priority_queue<dist_node_t, std::vector<dist_node_t>, CompareByFirst> PriorityQueue;

  // Large (several GB), pre-allocated block of memory.
  char* _index_memory;

  size_t _M;
  // size of one data point (does not support variable-size data, strings)
  size_t _data_size_bytes;
  // Node consists of: ([data] [M links] [data label]). This layout was chosen
  // after benchmarking - it's slightly more cache-efficient than others.
  size_t _node_size_bytes;
  size_t _max_node_count;  // Determines size of internal pre-allocated memory
  size_t _cur_num_nodes;
  std::unique_ptr<DistanceInterface<dist_t>> _distance;
  std::mutex _index_data_guard;

  uint32_t _num_threads;

  // Remembers which nodes we've visited, to avoid re-computing distances.
  VisitedSetPool* _visited_set_pool;
  std::vector<std::mutex> _node_links_mutexes;

  bool _collect_stats = false;
  DataType _data_type;

  // NOTE: These metrics are meaningful the most with single-threaded search.
  // With multi-threaded search, for instance, the number of distance computations will
  // accumulate across queries, which means at the end of the batched search, the number
  // you get is the cumulative sum of all distance computations across all queries.
  // Maybe that's what you want, but it's worth noting
  mutable std::atomic<uint64_t> _distance_computations = 0;
  mutable std::atomic<uint64_t> _metric_hops = 0;

  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;

  // A custom move constructor is needed because the class manages dynamic
  // resources (_index_memory, _visited_set_pool),
  // which require explicit ownership transfer and cleanup to avoid resource
  // leaks or double frees. The default move constructor cannot ensure these
  // resources are safely transferred and the source object is left in a valid
  // state.
  Index(Index&& other) noexcept
      : _index_memory(other._index_memory),
        _M(other._M),
        _data_size_bytes(other._data_size_bytes),
        _node_size_bytes(other._node_size_bytes),
        _max_node_count(other._max_node_count),
        _cur_num_nodes(other._cur_num_nodes),
        _distance(std::move(other._distance)),
        _index_data_guard(std::move(other._index_data_guard)),
        _num_threads(other._num_threads),
        _visited_set_pool(std::move(other._visited_set_pool)),
        _node_links_mutexes(std::move(other._node_links_mutexes)) {
    other._index_memory = nullptr;
    other._visited_set_pool = nullptr;
  }

  Index& operator=(Index&& other) noexcept {
    if (this != &other) {
      delete[] _index_memory;
      delete _visited_set_pool;

      _index_memory = other._index_memory;
      _M = other._M;
      _data_size_bytes = other._data_size_bytes;
      _node_size_bytes = other._node_size_bytes;
      _max_node_count = other._max_node_count;
      _cur_num_nodes = other._cur_num_nodes;
      _distance = std::move(other._distance);
      _index_data_guard = std::move(other._index_data_guard);
      _num_threads = other._num_threads;
      _visited_set_pool = std::move(other._visited_set_pool);
      _node_links_mutexes = std::move(other._node_links_mutexes);

      other._index_memory = nullptr;
      other._visited_set_pool = nullptr;
    }
    return *this;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_data_type, _M, _data_size_bytes, _node_size_bytes, _max_node_count, _cur_num_nodes, *_distance);

    // Serialize the allocated memory for the index & query.
    uint64_t total_mem = static_cast<uint64_t>(_node_size_bytes) * static_cast<uint64_t>(_max_node_count);
    archive(cereal::binary_data(_index_memory, total_mem));
  }

 public:
  enum class PruningStrategy {
    HNSW_HEURISTIC,
    ALPHA_DIVERSITY,
    SSG,
    RNG  // For future implementation
  };

  PruningStrategy _pruning_strategy;
  float _alpha;
  float _angle_threshold;
  /**
   * @brief Construct a new Index object for approximate near neighbor search.
   *
   * This constructor initializes an Index object with the specified distance
   * metric, dataset size, and maximum number of links per node. It also allows
   * for collecting statistics during the search process.
   *
   * @param dist The distance metric for the index. Options include l2
   * (euclidean) and inner product.
   * @param dataset_size The maximum number of vectors that can be inserted in
   * the index.
   * @param max_edges_per_node The maximum number of links per node.
   * @param collect_stats Flag indicating whether to collect statistics during
   * the search process.
   */
  Index(std::unique_ptr<DistanceInterface<dist_t>> dist, int dataset_size, int max_edges_per_node,
        bool collect_stats = false, DataType data_type = DataType::float32,
        PruningStrategy strategy = PruningStrategy::HNSW_HEURISTIC, float alpha = 1.2f,
        float angle_threshold = 60.0f)
      : _M(max_edges_per_node),
        _max_node_count(dataset_size),
        _cur_num_nodes(0),
        _distance(std::move(dist)),
        _num_threads(1),
        _visited_set_pool(new VisitedSetPool(
            /* initial_pool_size = */ 1,
            /* num_elements = */ dataset_size)),
        _node_links_mutexes(dataset_size),
        _collect_stats(collect_stats),
        _data_type(data_type),
        _pruning_strategy(strategy),
        _alpha(alpha),
        _angle_threshold(angle_threshold),
        _track_edges(false) {

    // Get the size in bytes of the _node_links_mutexes vector.
    size_t mutexes_size_bytes = _node_links_mutexes.size() * sizeof(std::mutex);

    _data_size_bytes = _distance->dataSize();
    _node_size_bytes = _data_size_bytes + (sizeof(node_id_t) * _M) + sizeof(label_t);
    uint64_t index_size = static_cast<uint64_t>(_node_size_bytes) * static_cast<uint64_t>(_max_node_count);
    _index_memory = new char[index_size];
  }

  ~Index() {
    delete[] _index_memory;
    delete _visited_set_pool;
  }

  void buildGraphLinks(const std::string& mtx_filename) {
    std::ifstream input_file(mtx_filename);
    if (!input_file.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + mtx_filename);
    }

    std::string line;
    // Skip the header
    while (std::getline(input_file, line)) {
      if (line[0] != '%')
        break;
    }

    std::istringstream iss(line);
    int num_vertices, num_edges;
    iss >> num_vertices >> num_vertices >> num_edges;

    // check that the number of vertices in the mtx file matches the number of
    // nodes in the index and that the number of edges is equal to the number of
    // links per node.
    if (num_vertices != _max_node_count) {
      throw std::runtime_error(
          "Number of vertices in the mtx file does not "
          "match the size allocated for the index.");
    }

    if (num_edges != _M) {
      throw std::runtime_error(
          "Number of edges in the mtx file does not match "
          "the number of links per node.");
    }

    int u, v;
    while (input_file >> u >> v) {
      // Adjust for 1-based indexing in Matrix Market format
      u--;
      v--;
      node_id_t* links = getNodeLinks(u);
      // Now add a directed edge from u to v. We need to check for the first
      // available slot in the links array since there might be other edges
      // added before this one. By definition, a slot is available if and only
      // if it points to the node itself.
      for (size_t i = 0; i < _M; i++) {
        if (links[i] == u) {
          links[i] = v;
          break;
        }
      }
    }

    input_file.close();
  }

  void enableEdgeTracking() {
    _track_edges = true;

    // Count total edges
    size_t total_edges = 0;
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t* links = getNodeLinks(node);
      for (size_t i = 0; i < _M; i++) {
        if (links[i] != node) {
          total_edges++;
        }
      }
    }

    // Initialize visit counts (no longer atomic, so resize works)
    _edge_visit_counts.clear();
    _edge_visit_counts.resize(total_edges, 0);  // Now works!

    // Build edge index map
    _edge_index_map.clear();
    size_t edge_idx = 0;
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t* links = getNodeLinks(node);
      for (size_t i = 0; i < _M; i++) {
        if (links[i] != node) {
          uint64_t edge_id = getEdgeId(node, links[i]);
          _edge_index_map[edge_id] = edge_idx++;
        }
      }
    }

    std::cout << "Edge tracking enabled. Tracking " << total_edges << " edges." << std::endl;
  }

  void disableEdgeTracking() { _track_edges = false; }

  struct EdgeUsageStats {
    size_t total_edges;
    size_t edges_with_visits;
    size_t edges_never_used;
    uint64_t total_visits;
    double avg_visits_per_edge;
    std::vector<std::pair<uint64_t, size_t>> top_edges;  // (visit_count, edge_idx)
  };

  EdgeUsageStats getEdgeUsageStats(int top_k = 10) const {
    EdgeUsageStats stats;
    stats.total_edges = _edge_visit_counts.size();
    stats.edges_with_visits = 0;
    stats.total_visits = 0;

    std::vector<std::pair<uint64_t, size_t>> edge_visits;

    for (size_t i = 0; i < _edge_visit_counts.size(); i++) {
      uint64_t visits = _edge_visit_counts[i];  // No .load() needed
      stats.total_visits += visits;

      if (visits > 0) {
        stats.edges_with_visits++;
        edge_visits.push_back({visits, i});
      }
    }

    stats.edges_never_used = stats.total_edges - stats.edges_with_visits;
    stats.avg_visits_per_edge =
        stats.total_edges > 0 ? static_cast<double>(stats.total_visits) / stats.total_edges : 0.0;

    // Get top-k
    std::partial_sort(edge_visits.begin(), edge_visits.begin() + std::min(top_k, (int)edge_visits.size()),
                      edge_visits.end(), std::greater<std::pair<uint64_t, size_t>>());

    for (int i = 0; i < std::min(top_k, (int)edge_visits.size()); i++) {
      stats.top_edges.push_back(edge_visits[i]);
    }

    return stats;
  }

  void pruneUnusedEdges(float keep_ratio) {
    if (!_track_edges || _edge_visit_counts.empty()) {
      throw std::runtime_error("Edge tracking must be enabled before pruning");
    }

    if (keep_ratio <= 0.0f || keep_ratio > 1.0f) {
      throw std::invalid_argument("keep_ratio must be between 0 and 1");
    }

    std::cout << "Pruning edges with keep_ratio=" << keep_ratio << std::endl;

    // PER-NODE pruning to maintain connectivity
    size_t pruned_count = 0;
    size_t min_edges_per_node = std::max(1, static_cast<int>(_M * keep_ratio));

    std::cout << "Minimum edges per node: " << min_edges_per_node << " (out of " << _M << ")" << std::endl;

    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t* links = getNodeLinks(node);

      // Collect (visit_count, edge_id, link_index) for this node's edges
      std::vector<std::tuple<uint64_t, uint64_t, size_t>> node_edges;

      for (size_t i = 0; i < _M; i++) {
        if (links[i] != node) {
          uint64_t edge_id = getEdgeId(node, links[i]);
          auto it = _edge_index_map.find(edge_id);

          if (it != _edge_index_map.end()) {
            uint64_t visits = _edge_visit_counts[it->second];
            node_edges.push_back({visits, edge_id, i});
          }
        }
      }

      // Sort this node's edges by visit count (descending)
      std::sort(node_edges.begin(), node_edges.end(),
                [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

      // Keep top edges per node (at least min_edges_per_node)
      size_t keep_for_this_node =
          std::max(min_edges_per_node, static_cast<size_t>(node_edges.size() * keep_ratio));

      // Mark edges to prune (those beyond keep_for_this_node)
      for (size_t j = keep_for_this_node; j < node_edges.size(); j++) {
        size_t link_idx = std::get<2>(node_edges[j]);
        links[link_idx] = node;  // Replace with self-loop
        pruned_count++;
      }
    }

    std::cout << "Pruned " << pruned_count << " edges while maintaining connectivity" << std::endl;
  }

  void setAngleThreshold(float threshold) {
    if (threshold <= 0.0f || threshold > 180.0f) {
      throw std::invalid_argument("Angle threshold must be between 0 and 180 degrees");
    }
    _angle_threshold = threshold;
  }

  float getAngleThreshold() const { return _angle_threshold; }

  std::vector<std::vector<uint32_t>> getGraphOutdegreeTable() {
    std::vector<std::vector<uint32_t>> outdegree_table(_cur_num_nodes);
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t* links = getNodeLinks(node);
      for (int i = 0; i < _M; i++) {
        if (links[i] != node) {
          outdegree_table[node].push_back(links[i]);
        }
      }
    }
    return outdegree_table;
  }

  size_t countActualEdges() const {
    size_t total_edges = 0;
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t* links = getNodeLinks(node);
      for (size_t i = 0; i < _M; i++) {
        if (links[i] != node) {
          total_edges++;
        }
      }
    }
    return total_edges;
  }

  float getAverageOutDegree() const {
    if (_cur_num_nodes == 0)
      return 0.0f;
    return static_cast<float>(countActualEdges()) / static_cast<float>(_cur_num_nodes);
  }

  void getEdgeStatistics() const {
    size_t total_edges = countActualEdges();
    size_t max_possible = _cur_num_nodes * _M;
    float avg_degree = getAverageOutDegree();
    float utilization = static_cast<float>(total_edges) / static_cast<float>(max_possible) * 100.0f;

    std::cout << "\nEdge Statistics\n" << std::flush;
    std::cout << "-----------------------------\n" << std::flush;
    std::cout << "Total actual edges: " << total_edges << "\n" << std::flush;
    std::cout << "Max possible edges: " << max_possible << "\n" << std::flush;
    std::cout << "Average out-degree: " << avg_degree << " / " << _M << "\n" << std::flush;
    std::cout << "Edge utilization: " << utilization << "%\n" << std::flush;
  }

  // Hub stuff

  /**
   * @brief Enable hub-aware construction with differential M values
   * 
   * When enabled, hub nodes will be given more edges (M_hub) while
   * feeder nodes get fewer edges (M_feeder). This creates an explicit
   * hub-feeder architecture that maintains the hub highway structure.
   * 
   * Must be called BEFORE identifying hubs and BEFORE construction.
   * 
   * @param M_hub Maximum edges for hub nodes (default: 2*M)
   * @param M_feeder Maximum edges for feeder nodes (default: M)
   */
  void enableHubAwareConstruction(size_t M_hub = 0, size_t M_feeder = 0) {
    if (M_hub == 0) {
      M_hub = _M * 2;  // Default: double the edges for hubs
    }
    if (M_feeder == 0) {
      M_feeder = _M;  // Default: standard M for feeders
    }

    if (M_hub < M_feeder) {
      throw std::invalid_argument("M_hub must be >= M_feeder");
    }

    if (M_hub > _M) {
      throw std::runtime_error(
          "M_hub exceeds allocated space. Must reconstruct index with larger max_edges_per_node.");
    }

    _hub_aware_construction = true;
    _M_hub = M_hub;
    _M_feeder = M_feeder;

    std::cout << "Hub-aware construction enabled:\n";
    std::cout << "  M_hub = " << _M_hub << "\n";
    std::cout << "  M_feeder = " << _M_feeder << "\n";
    std::cout << "  Note: Hubs must be identified after construction via identifyHubsByExtrema()\n";
  }

  void disableHubAwareConstruction() { _hub_aware_construction = false; }

  bool isHubAwareConstructionEnabled() const { return _hub_aware_construction; }

  /**
   * @brief Identify hub nodes based on distance to centroid.
   * 
   * Hubs in high-dimensional spaces tend to be at the extrema of the dataset,
   * which manifests as points far from the centroid. This method computes
   * the centroid of all nodes and marks the top percentile as hubs.
   * 
   * @param hub_percentile Percentile threshold (95 = top 5% are hubs)
   */
  void identifyHubsByExtrema(float hub_percentile = 95.0f) {
    if (_cur_num_nodes == 0) {
      throw std::runtime_error("Cannot identify hubs on empty index");
    }

    if (hub_percentile <= 0.0f || hub_percentile >= 100.0f) {
      throw std::invalid_argument("hub_percentile must be between 0 and 100");
    }

    _hub_percentile = hub_percentile;

    std::cout << "Computing dataset centroid..." << std::flush;

    // Step 1: Compute centroid
    size_t dim = _distance->dimension();
    std::vector<float> centroid(dim, 0.0f);

    // Sum all vectors
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      float* node_data = reinterpret_cast<float*>(getNodeData(node));
      for (size_t d = 0; d < dim; d++) {
        centroid[d] += node_data[d];
      }
    }

    // Average
    for (size_t d = 0; d < dim; d++) {
      centroid[d] /= static_cast<float>(_cur_num_nodes);
    }

    std::cout << " done." << std::endl;
    std::cout << "Computing distances to centroid..." << std::flush;

    // Step 2: Compute distance from each node to centroid
    _hub_scores.resize(_cur_num_nodes);
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      _hub_scores[node] = _distance->distance(
          /* x = */ centroid.data(),
          /* y = */ getNodeData(node),
          /* asymmetric = */ false);
    }

    std::cout << " done." << std::endl;

    // Step 3: Determine threshold
    std::vector<float> sorted_scores = _hub_scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());

    size_t threshold_idx = static_cast<size_t>((hub_percentile / 100.0f) * _cur_num_nodes);
    float threshold = sorted_scores[threshold_idx];

    // Step 4: Mark hubs
    _hub_mask.resize(_cur_num_nodes);
    size_t hub_count = 0;
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      _hub_mask[node] = (_hub_scores[node] >= threshold);
      if (_hub_mask[node]) {
        hub_count++;
      }
    }

    _hubs_identified = true;

    std::cout << "Identified " << hub_count << " hub nodes (" << (100.0f * hub_count / _cur_num_nodes)
              << "% of dataset)" << std::endl;
    std::cout << "Hub threshold distance: " << threshold << std::endl;
  }

  /**
   * @brief Pre-identify hubs BEFORE construction starts
   * 
   * This computes the centroid from raw data and marks which nodes will be hubs.
   * Must be called BEFORE any nodes are added to the index.
   * 
   * @param data Raw dataset pointer
   * @param num_points Number of points in dataset
   * @param hub_percentile Percentile threshold (95 = top 5% are hubs)
   */
  void preIdentifyHubs(const void* data, size_t num_points, float hub_percentile = 95.0f) {
    if (_cur_num_nodes > 0) {
      throw std::runtime_error("preIdentifyHubs must be called before adding any nodes");
    }

    if (hub_percentile <= 0.0f || hub_percentile >= 100.0f) {
      throw std::invalid_argument("hub_percentile must be between 0 and 100");
    }

    _hub_percentile = hub_percentile;

    std::cout << "Pre-identifying hubs from raw data..." << std::flush;

    // Step 1: Compute centroid from raw data
    size_t dim = _distance->dimension();
    std::vector<float> centroid(dim, 0.0f);

    const float* float_data = reinterpret_cast<const float*>(data);

    for (size_t point = 0; point < num_points; point++) {
      for (size_t d = 0; d < dim; d++) {
        centroid[d] += float_data[point * dim + d];
      }
    }

    for (size_t d = 0; d < dim; d++) {
      centroid[d] /= static_cast<float>(num_points);
    }

    std::cout << " computing distances..." << std::flush;

    // Step 2: Compute distances to centroid
    _hub_scores.resize(num_points);
    for (size_t point = 0; point < num_points; point++) {
      const void* point_data = &float_data[point * dim];
      _hub_scores[point] = _distance->distance(centroid.data(), point_data, false);
    }

    std::cout << " marking hubs..." << std::flush;

    // Step 3: Determine threshold
    std::vector<float> sorted_scores = _hub_scores;
    std::sort(sorted_scores.begin(), sorted_scores.end());

    size_t threshold_idx = static_cast<size_t>((hub_percentile / 100.0f) * num_points);
    float threshold = sorted_scores[threshold_idx];

    // Step 4: Mark hubs
    _hub_mask.resize(num_points);
    size_t hub_count = 0;
    for (size_t point = 0; point < num_points; point++) {
      _hub_mask[point] = (_hub_scores[point] >= threshold);
      if (_hub_mask[point]) {
        hub_count++;
      }
    }

    _hubs_identified = true;

    std::cout << " done." << std::endl;
    std::cout << "Pre-identified " << hub_count << " hub nodes (" << (100.0f * hub_count / num_points)
              << "% of dataset)" << std::endl;
    std::cout << "Hub threshold distance: " << threshold << std::endl;
  }

  /**
   * @brief Check if a node is a hub
   */
  bool isHub(node_id_t node_id) const {
    if (!_hubs_identified) {
      throw std::runtime_error(
          "Hubs must be identified before querying. Call identifyHubsByExtrema() first.");
    }
    if (node_id >= _cur_num_nodes) {
      throw std::out_of_range("Node ID out of range");
    }
    return _hub_mask[node_id];
  }

  /**
   * @brief Get hub score (distance to centroid) for a node
   */
  float getHubScore(node_id_t node_id) const {
    if (!_hubs_identified) {
      throw std::runtime_error(
          "Hubs must be identified before querying. Call identifyHubsByExtrema() first.");
    }
    if (node_id >= _cur_num_nodes) {
      throw std::out_of_range("Node ID out of range");
    }
    return _hub_scores[node_id];
  }

  /**
   * @brief Get vector of hub node IDs
   */
  std::vector<node_id_t> getHubNodeIds() const {
    if (!_hubs_identified) {
      throw std::runtime_error(
          "Hubs must be identified before querying. Call identifyHubsByExtrema() first.");
    }

    std::vector<node_id_t> hub_ids;
    hub_ids.reserve(_cur_num_nodes * (100.0f - _hub_percentile) / 100.0f);

    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      if (_hub_mask[node]) {
        hub_ids.push_back(node);
      }
    }

    return hub_ids;
  }

  /**
   * @brief Get hub statistics
   */
  void getHubStatistics() const {
    if (!_hubs_identified) {
      throw std::runtime_error("Hubs must be identified first. Call identifyHubsByExtrema().");
    }

    size_t hub_count = 0;
    float min_hub_score = std::numeric_limits<float>::max();
    float max_hub_score = std::numeric_limits<float>::lowest();
    float avg_hub_score = 0.0f;

    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      if (_hub_mask[node]) {
        hub_count++;
        float score = _hub_scores[node];
        min_hub_score = std::min(min_hub_score, score);
        max_hub_score = std::max(max_hub_score, score);
        avg_hub_score += score;
      }
    }

    if (hub_count > 0) {
      avg_hub_score /= static_cast<float>(hub_count);
    }

    std::cout << "\nHub Statistics\n";
    std::cout << "-----------------------------\n";
    std::cout << "Total hubs: " << hub_count << " / " << _cur_num_nodes << " ("
              << (100.0f * hub_count / _cur_num_nodes) << "%)\n";
    std::cout << "Hub percentile threshold: " << _hub_percentile << "\n";
    std::cout << "Hub score range: [" << min_hub_score << ", " << max_hub_score << "]\n";
    std::cout << "Average hub score: " << avg_hub_score << "\n";
    std::cout << std::flush;
  }

  /**
   * @brief Reset hub identification (useful for re-running with different parameters)
   */
  void resetHubIdentification() {
    _hubs_identified = false;
    _hub_mask.clear();
    _hub_scores.clear();
  }

  /**
   * @brief Get hub connectivity statistics
   * Shows how well-connected hubs are compared to non-hubs
   */
  void getHubConnectivityStats() const {
    if (!_hubs_identified) {
      throw std::runtime_error("Hubs must be identified first.");
    }

    size_t hub_to_hub_edges = 0;
    size_t hub_to_feeder_edges = 0;
    size_t feeder_to_hub_edges = 0;
    size_t feeder_to_feeder_edges = 0;

    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      bool node_is_hub = _hub_mask[node];
      node_id_t* links = getNodeLinks(node);

      for (size_t i = 0; i < _M; i++) {
        node_id_t neighbor = links[i];
        if (neighbor == node)
          continue;  // Skip self-loops

        bool neighbor_is_hub = _hub_mask[neighbor];

        if (node_is_hub && neighbor_is_hub) {
          hub_to_hub_edges++;
        } else if (node_is_hub && !neighbor_is_hub) {
          hub_to_feeder_edges++;
        } else if (!node_is_hub && neighbor_is_hub) {
          feeder_to_hub_edges++;
        } else {
          feeder_to_feeder_edges++;
        }
      }
    }

    size_t total_edges =
        hub_to_hub_edges + hub_to_feeder_edges + feeder_to_hub_edges + feeder_to_feeder_edges;

    std::cout << "\nHub Connectivity Statistics\n";
    std::cout << "-----------------------------\n";
    std::cout << "Hub→Hub edges: " << hub_to_hub_edges << " (" << (100.0f * hub_to_hub_edges / total_edges)
              << "%)\n";
    std::cout << "Hub→Feeder edges: " << hub_to_feeder_edges << " ("
              << (100.0f * hub_to_feeder_edges / total_edges) << "%)\n";
    std::cout << "Feeder→Hub edges: " << feeder_to_hub_edges << " ("
              << (100.0f * feeder_to_hub_edges / total_edges) << "%)\n";
    std::cout << "Feeder→Feeder edges: " << feeder_to_feeder_edges << " ("
              << (100.0f * feeder_to_feeder_edges / total_edges) << "%)\n";
    std::cout << std::flush;
  }

  /**
   * @brief Store the new node in the global data structure. In a
   * multi-threaded setting, the index data guard should be held by the caller
   * with an exclusive lock.
   *
   * @param data The vector to add.
   * @param label The label (meta-data) of the vector.
   * @param new_node_id The id of the new node.
   */
  void allocateNode(void* data, label_t& label, node_id_t& new_node_id) {
    new_node_id = _cur_num_nodes;
    _distance->transformData(
        /* destination = */ getNodeData(new_node_id),
        /* src = */ data);
    *(getNodeLabel(new_node_id)) = label;
    node_id_t* links = getNodeLinks(new_node_id);
    // Initialize all edges to self
    std::fill_n(links, _M, new_node_id);
    _cur_num_nodes++;
  }

  /**
   * @brief Adds vectors to the index in batches.
   *
   * This method is responsible for adding vectors in batches, represented by
   * `data`, to the underlying graph. Each vector is associated with a label
   * provided in the `labels` vector. The method efficiently handles concurrent
   * additions by dividing the workload among multiple threads, defined by
   * `_num_threads`.
   *
   * The method ensures thread safety by employing locking mechanisms at the
   * node level in the underlying `connectNeighbors` and `beamSearch` methods.
   * This allows multiple threads to safely add vectors to the index without
   * causing data races or inconsistencies in the graph structure.
   *
   * @param data Pointer to the array of vectors to be added.
   * @param labels A vector of labels corresponding to each vector in `data`.
   * @param ef_construction Parameter for controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm. Must be greater than 0.
   *
   * @exception std::invalid_argument Thrown if `num_initializations` is less
   * than or equal to 0.
   * @exception std::runtime_error Thrown if the maximum number of nodes in the
   * index is reached.
   */
  template <typename data_type>
  void addBatch(void* data, std::vector<label_t>& labels, int ef_construction,
                int num_initializations = 100) {
    if (num_initializations <= 0) {
      throw std::invalid_argument("num_initializations must be greater than 0.");
    }
    uint32_t total_num_nodes = labels.size();
    uint32_t data_dimension = _distance->dimension();

    // Don't spawn any threads if we are only using one.
    if (_num_threads == 1) {
      for (uint32_t row_index = 0; row_index < total_num_nodes; row_index++) {
        uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
        void* vector = (data_type*)data + offset;
        label_t label = labels[row_index];
        this->add(vector, label, ef_construction, num_initializations);
      }
      return;
    }

    flatnav::executeInParallel(
        /* start_index = */ 0, /* end_index = */ total_num_nodes,
        /* num_threads = */ _num_threads, /* function = */
        [&](uint32_t row_index) {
          uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
          void* vector = (data_type*)data + offset;
          label_t label = labels[row_index];
          this->add(vector, label, ef_construction, num_initializations);
        });
  }

  /**
   * @brief Adds a single vector to the index.
   *
   * This method is called internally by `addBatch` for each vector in the
   * batch. The method ensures thread safety by using locking primitives,
   * allowing it to be safely used in a multi-threaded environment.
   *
   * The method first checks if the current number of nodes has reached the
   * maximum capacity. If so, it throws a runtime error. It then locks the index
   * structure to prevent concurrent modifications while allocating a new node.
   * After unlocking, it connects the new node to its neighbors in the graph.
   *
   * @param data Pointer to the vector data being added.
   * @param label Label associated with the vector.
   * @param ef_construction Parameter controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm.
   *
   * @exception std::runtime_error Thrown if the maximum number of nodes is
   * reached.
   */
  void add(void* data, label_t& label, int ef_construction, int num_initializations) {
    if (_cur_num_nodes >= _max_node_count) {
      throw std::runtime_error(
          "Maximum number of nodes reached. Consider "
          "increasing the `max_node_count` parameter to "
          "create a larger index.");
    }

    std::unique_lock<std::mutex> global_lock(_index_data_guard);
    auto entry_node = initializeSearch(data, num_initializations);
    node_id_t new_node_id;

    bool is_predicted_hub = false;
    if (_hub_aware_construction && _hubs_identified) {
      // _cur_num_nodes is the index of the node we're about to add
      is_predicted_hub = (_cur_num_nodes < _hub_mask.size()) && _hub_mask[_cur_num_nodes];
    }

    allocateNode(data, label, new_node_id);
    global_lock.unlock();

    if (new_node_id == 0) {
      return;
    }

    int adjusted_ef_construction = ef_construction;
    if (_hub_aware_construction && _hubs_identified && is_predicted_hub) {
      // Give hubs 3x the search budget to find other hubs
      adjusted_ef_construction = std::min(ef_construction * 3, static_cast<int>(_max_node_count / 10));
    }

    auto neighbors = beamSearch(
        /* query = */ data,
        /* entry_node = */ entry_node,
        /* buffer_size = */ adjusted_ef_construction);

    // Determine M based on hub status (if hub-aware is enabled and hubs identified)
    int selection_M;
    if (_hub_aware_construction && _hubs_identified) {
      // Check if this node will be a hub (predict based on centroid distance)
      // We need to predict hub status for the new node
      bool predicted_hub = predictHubStatus(data);
      selection_M = predicted_hub ? std::max(static_cast<int>(_M_hub / 2), 1)
                                  : std::max(static_cast<int>(_M_feeder / 2), 1);
    } else {
      selection_M = std::max(static_cast<int>(_M / 2), 1);
    }

    selectNeighborsUnified(/* neighbors = */ neighbors, /* M = */ selection_M,
                           /* query = */ data, /* node_id = */ new_node_id);
    connectNeighbors(neighbors, new_node_id);
  }

  /***
   * @brief Search the index for the k nearest neighbors of the query.
   * @param query The query vector.
   * @param K The number of nearest neighbors to return.
   * @param ef_search The search beam width.
   * @param num_initializations The number of random initializations to use.
   */
  std::vector<dist_label_t> search(const void* query, const int K, int ef_search,
                                   int num_initializations = 100) {
    node_id_t entry_node = initializeSearch(query, num_initializations);
    PriorityQueue neighbors = beamSearch(/* query = */ query,
                                         /* entry_node = */ entry_node,
                                         /* buffer_size = */ std::max(ef_search, K));
    auto size = neighbors.size();
    std::vector<dist_label_t> results;
    results.reserve(size);
    while (!neighbors.empty()) {
      auto [distance, node_id] = neighbors.top();
      auto label = *getNodeLabel(node_id);
      results.emplace_back(distance, label);
      neighbors.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const dist_label_t& left, const dist_label_t& right) { return left.first < right.first; });
    if (results.size() > static_cast<size_t>(K)) {
      results.resize(K);
    }

    return results;
  }

  void doGraphReordering(const std::vector<std::string>& reordering_methods) {

    for (const auto& method : reordering_methods) {
      auto outdegree_table = getGraphOutdegreeTable();
      std::vector<node_id_t> P;
      if (method == "gorder") {
        P = std::move(util::gOrder<node_id_t>(outdegree_table, 5));
      } else if (method == "rcm") {
        P = std::move(util::rcmOrder<node_id_t>(outdegree_table));
      } else {
        throw std::invalid_argument("Invalid reordering method: " + method);
      }

      relabel(P);
    }
  }

  void reorderGOrder(const int window_size = 5) {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = util::gOrder<node_id_t>(outdegree_table, window_size);

    relabel(P);
  }

  void reorderRCM() {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = util::rcmOrder<node_id_t>(outdegree_table);
    relabel(P);
  }

  static std::unique_ptr<Index<dist_t, label_t>> loadIndex(const std::string& filename) {
    std::ifstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    cereal::BinaryInputArchive archive(stream);
    std::unique_ptr<Index<dist_t, label_t>> index(new Index<dist_t, label_t>());

    std::unique_ptr<DistanceInterface<dist_t>> dist = std::make_unique<dist_t>();

    // 1. Deserialize metadata
    archive(index->_data_type, index->_M, index->_data_size_bytes, index->_node_size_bytes,
            index->_max_node_count, index->_cur_num_nodes, *dist);
    index->_visited_set_pool = new VisitedSetPool(
        /* initial_pool_size = */ 1,
        /* num_elements = */ index->_max_node_count);
    index->_distance = std::move(dist);
    index->_num_threads = std::max((uint32_t)1, (uint32_t)std::thread::hardware_concurrency() / 2);
    index->_node_links_mutexes = std::vector<std::mutex>(index->_max_node_count);

    // 2. Allocate memory using deserialized metadata
    uint64_t mem_size =
        static_cast<uint64_t>(index->_node_size_bytes) * static_cast<uint64_t>(index->_max_node_count);

    index->_index_memory = new char[mem_size];

    // 3. Deserialize content into allocated memory
    archive(cereal::binary_data(index->_index_memory, mem_size));

    return index;
  }

  void saveIndex(const std::string& filename) {
    std::ofstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    cereal::BinaryOutputArchive archive(stream);
    archive(*this);
  }

  inline void setNumThreads(uint32_t num_threads) {
    if (num_threads == 0 || num_threads > std::thread::hardware_concurrency()) {
      throw std::invalid_argument(
          "Number of threads must be greater than 0 and less than or equal to "
          "the number of hardware threads.");
    }
    _num_threads = num_threads;
    if (_num_threads == 1) {
      _visited_set_pool->setPoolSize(1);
    }
  }

  inline uint64_t getTotalIndexMemory() const {
    return static_cast<uint64_t>(_node_size_bytes) * static_cast<uint64_t>(_max_node_count);
  }
  inline uint64_t mutexesAllocatedMemory() const {
    return static_cast<uint64_t>(_node_links_mutexes.size() * sizeof(std::mutex));
  }

  inline uint64_t visitedSetPoolAllocatedMemory() const {
    size_t pool_size = _visited_set_pool->poolSize();
    return static_cast<uint64_t>(pool_size * sizeof(VisitedSet));
  }

  inline uint32_t getNumThreads() const { return _num_threads; }

  inline size_t maxEdgesPerNode() const { return _M; }
  inline size_t dataSizeBytes() const { return _data_size_bytes; }

  inline size_t nodeSizeBytes() const { return _node_size_bytes; }

  inline size_t maxNodeCount() const { return _max_node_count; }

  inline size_t currentNumNodes() const { return _cur_num_nodes; }
  inline size_t dataDimension() const { return _distance->dimension(); }

  inline uint64_t distanceComputations() const { return _distance_computations.load(); }

  inline DataType getDataType() const { return _data_type; }

  void resetStats() {
    _distance_computations = 0;
    _metric_hops = 0;
  }

  void getIndexSummary() const {
    std::cout << "\nIndex Parameters\n" << std::flush;
    std::cout << "-----------------------------\n" << std::flush;
    std::cout << "max_edges_per_node (M): " << _M << "\n" << std::flush;
    std::cout << "data_size_bytes: " << _data_size_bytes << "\n" << std::flush;
    std::cout << "node_size_bytes: " << _node_size_bytes << "\n" << std::flush;
    std::cout << "max_node_count: " << _max_node_count << "\n" << std::flush;
    std::cout << "cur_num_nodes: " << _cur_num_nodes << "\n" << std::flush;

    _distance->getSummary();
  }

  void setPruningStrategy(PruningStrategy strategy) { _pruning_strategy = strategy; }

  void setAlpha(float alpha) {
    if (alpha <= 0.0f) {
      throw std::invalid_argument("Alpha must be positive");
    }
    _alpha = alpha;
  }

  PruningStrategy getPruningStrategy() const { return _pruning_strategy; }

  float getAlpha() const { return _alpha; }

 private:
  friend class cereal::access;
  // Default constructor for cereal
  Index() = default;

  float computeAngle(const void* vec1, const void* vec2, const void* center) const {
    // Compute vectors from center to vec1 and vec2
    // angle = arccos(dot(v1-c, v2-c) / (norm(v1-c) * norm(v2-c)))

    // For L2 distance, we can use the law of cosines:
    // cos(angle) = (d1² + d2² - d12²) / (2 * d1 * d2)
    // where d1 = dist(center, vec1), d2 = dist(center, vec2), d12 = dist(vec1, vec2)

    float d1 = _distance->distance(center, vec1, false);
    float d2 = _distance->distance(center, vec2, false);
    float d12 = _distance->distance(vec1, vec2, false);

    if (d1 < 1e-10f || d2 < 1e-10f)
      return 0.0f;  // Avoid division by zero

    float cos_angle = (d1 * d1 + d2 * d2 - d12 * d12) / (2.0f * d1 * d2);

    // Clamp to [-1, 1] to handle numerical errors
    cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));

    // Convert to degrees
    float angle_rad = std::acos(cos_angle);
    float angle_deg = angle_rad * 180.0f / M_PI;

    return angle_deg;
  }

  bool _track_edges;
  std::vector<uint64_t> _edge_visit_counts;  // Change from atomic<uint64_t>
  std::unordered_map<uint64_t, size_t> _edge_index_map;
  std::mutex _edge_tracking_mutex;

  // Hub stuff
  std::vector<bool> _hub_mask;
  std::vector<float> _hub_scores;
  bool _hubs_identified = false;
  float _hub_percentile = 95.0f;

  bool _hub_aware_construction = false;
  size_t _M_hub = 32;  // Max edges for hub nodes
  size_t _M_feeder = 16;

  /**
   * @brief Hub-aware neighbor selection strategy (AGGRESSIVE VERSION - FIXED)
   */
  void selectNeighborsHubAware(PriorityQueue& neighbors, int M, const void* query, node_id_t node_id) {
    if (!_hubs_identified) {
      selectNeighborsUnified(neighbors, M, query);
      return;
    }

    if (neighbors.size() <= M) {
      return;
    }

    bool is_hub = _hub_mask[node_id];

    // Convert max-heap to min-heap (sort by distance ascending)
    std::priority_queue<std::pair<float, node_id_t>, std::vector<std::pair<float, node_id_t>>,
                        std::greater<std::pair<float, node_id_t>>>
        candidates;

    while (!neighbors.empty()) {
      auto [dist, id] = neighbors.top();
      candidates.emplace(dist, id);
      neighbors.pop();
    }

    std::vector<dist_node_t> selected;
    selected.reserve(M);

    if (is_hub) {
      // ===== HUB STRATEGY: AGGRESSIVELY BUILD HUB HIGHWAY =====

      std::vector<std::pair<float, node_id_t>> hub_candidates;
      std::vector<std::pair<float, node_id_t>> feeder_candidates;

      while (!candidates.empty()) {
        auto [dist, candidate] = candidates.top();
        candidates.pop();

        if (_hub_mask[candidate]) {
          hub_candidates.push_back({dist, candidate});
        } else {
          feeder_candidates.push_back({dist, candidate});
        }
      }

      // Phase 1: PRIORITIZE HUB CONNECTIONS (up to 75% of edges)
      int hub_target = static_cast<int>(M * 0.75);

      for (const auto& [dist, candidate] : hub_candidates) {
        if (selected.size() >= M)
          break;
        if (selected.size() >= hub_target)
          break;

        // Add hub connection without diversity check
        selected.push_back({dist, candidate});
      }

      // Phase 2: Fill remaining with closest feeders
      for (const auto& [dist, candidate] : feeder_candidates) {
        if (selected.size() >= M)
          break;

        // Apply diversity check for feeders
        bool keep = true;
        for (const auto& [_, already_selected] : selected) {
          float d_neighbor =
              _distance->distance(getNodeData(candidate), getNodeData(already_selected), false);

          if (d_neighbor < dist) {
            keep = false;
            break;
          }
        }

        if (keep) {
          selected.push_back({dist, candidate});
        }
      }

    } else {
      // ===== FEEDER STRATEGY: AGGRESSIVELY CONNECT TO HIGHWAY =====

      std::vector<std::pair<float, node_id_t>> hub_neighbors;
      std::vector<std::pair<float, node_id_t>> regular_neighbors;  // FIXED: was feeder_neighbors

      while (!candidates.empty()) {
        auto [dist, candidate] = candidates.top();
        candidates.pop();

        if (_hub_mask[candidate]) {
          hub_neighbors.push_back({dist, candidate});
        } else {
          regular_neighbors.push_back({dist, candidate});  // FIXED
        }
      }

      // Phase 1: MAXIMIZE hub connections (66% of edges)
      int hub_target = static_cast<int>(M * 0.66);

      for (size_t i = 0; i < hub_neighbors.size() && selected.size() < hub_target; i++) {
        selected.push_back(hub_neighbors[i]);
      }

      // Phase 2: Add diverse local connections
      for (const auto& [dist, candidate] : regular_neighbors) {  // FIXED
        if (selected.size() >= M)
          break;

        bool keep = true;
        for (const auto& [_, already_selected] : selected) {
          float d_neighbor =
              _distance->distance(getNodeData(candidate), getNodeData(already_selected), false);

          if (d_neighbor < dist) {
            keep = false;
            break;
          }
        }

        if (keep) {
          selected.push_back({dist, candidate});
        }
      }
    }

    // Put selected neighbors back into priority queue
    for (const auto& [dist, id] : selected) {
      neighbors.emplace(dist, id);
    }
  }

  /**
   * @brief Predict if a node will be a hub based on its distance to centroid
   * Used during construction to apply hub-aware strategies before full hub identification
   */
  bool predictHubStatus(const void* data) {
    if (!_hubs_identified || _hub_mask.empty()) {
      return false;  // Default to feeder if hubs not pre-identified
    }

    // During construction, _cur_num_nodes represents the node being added
    // We subtract 1 because allocateNode already incremented _cur_num_nodes
    node_id_t node_idx = _cur_num_nodes - 1;

    if (node_idx >= _hub_mask.size()) {
      return false;  // Safety check
    }

    return _hub_mask[node_idx];
  }

  // Helper to get unique edge identifier
  uint64_t getEdgeId(node_id_t from, node_id_t to) const {
    return (static_cast<uint64_t>(from) << 32) | static_cast<uint64_t>(to);
  }

  void trackEdgesForNode(node_id_t from_node, const VisitedSet* visited_set) {
    if (!_track_edges) {
      return;
    }

    node_id_t* links = getNodeLinks(from_node);

    for (size_t i = 0; i < _M; i++) {
      node_id_t to_node = links[i];

      // Skip self-loops and already visited nodes
      if (to_node == from_node || visited_set->isVisited(to_node)) {
        continue;
      }

      // Track this edge
      uint64_t edge_id = getEdgeId(from_node, to_node);
      auto it = _edge_index_map.find(edge_id);

      if (it != _edge_index_map.end()) {
        // Thread-safe increment
        std::lock_guard<std::mutex> lock(_edge_tracking_mutex);
        _edge_visit_counts[it->second]++;
      }
    }
  }

  /**
   * @brief RNG (Relative Neighborhood Graph) neighbor selection
   * 
   * For each candidate, keep it only if there's no already-selected neighbor
   * that is closer to both the query and the candidate than they are to each other.
   * 
   * Formally, keep edge (query, candidate) if for all selected neighbors s:
   *   dist(query, candidate) <= max(dist(query, s), dist(candidate, s))
   * 
   * This creates a more sparse but well-connected graph.
   */
  void selectNeighborsRNG(PriorityQueue& neighbors, int M, const void* query) {
    if (neighbors.size() <= M) {
      return;
    }

    // Convert max-heap to min-heap (sort by distance ascending)
    std::priority_queue<std::pair<float, node_id_t>, std::vector<std::pair<float, node_id_t>>,
                        std::greater<std::pair<float, node_id_t>>>
        candidates;

    while (!neighbors.empty()) {
      auto [dist, id] = neighbors.top();
      candidates.emplace(dist, id);
      neighbors.pop();
    }

    std::vector<dist_node_t> selected;
    selected.reserve(M);

    while (!candidates.empty() && selected.size() < M) {
      auto [d_query_candidate, candidate] = candidates.top();
      candidates.pop();

      bool keep = true;

      // Check RNG condition with all already selected neighbors
      for (const auto& [d_query_selected, already_selected] : selected) {
        // Distance between candidate and already_selected neighbor
        float d_candidate_selected =
            _distance->distance(getNodeData(candidate), getNodeData(already_selected), false);

        // RNG condition: prune if there exists a neighbor that's closer to both
        // the query and the candidate than they are to each other
        // Keep edge if: d(query,candidate) <= max(d(query,selected), d(candidate,selected))
        float max_dist = std::max(d_query_selected, d_candidate_selected);

        if (d_query_candidate > max_dist) {
          // There's a shortcut through already_selected - prune this candidate
          keep = false;
          break;
        }
      }

      if (keep) {
        selected.push_back({d_query_candidate, candidate});
      }
    }

    // Put selected neighbors back into the priority queue
    for (const auto& [dist, id] : selected) {
      neighbors.emplace(dist, id);
    }
  }

  void selectNeighborsSSG(PriorityQueue& neighbors, int M, const void* query) {
    if (neighbors.size() <= M) {
      return;
    }

    std::priority_queue<std::pair<float, node_id_t>, std::vector<std::pair<float, node_id_t>>,
                        std::greater<std::pair<float, node_id_t>>>
        candidates;

    while (!neighbors.empty()) {
      auto [dist, id] = neighbors.top();
      candidates.emplace(dist, id);
      neighbors.pop();
    }

    std::vector<dist_node_t> selected;
    selected.reserve(M);

    while (!candidates.empty() && selected.size() < M) {
      auto [d_query, candidate] = candidates.top();
      candidates.pop();

      bool keep = true;

      for (const auto& [_, already_selected] : selected) {
        float angle = computeAngle(getNodeData(candidate), getNodeData(already_selected), query);

        // USE COMPLEMENTARY ANGLE: how much the angle deviates from 180°
        // This measures "similarity" in direction rather than difference
        float angular_deviation = 180.0f - angle;

        // Prune if directions are too similar (small deviation from opposite)
        if (angular_deviation < _angle_threshold) {
          keep = false;
          break;
        }
      }

      if (keep) {
        selected.push_back({d_query, candidate});
      }
    }

    for (const auto& [dist, id] : selected) {
      neighbors.emplace(dist, id);
    }
  }

  char* getNodeData(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_node_size_bytes);
    return _index_memory + byte_offset;
  }

  node_id_t* getNodeLinks(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_node_size_bytes);
    byte_offset += _data_size_bytes;
    char* location = _index_memory + byte_offset;
    return reinterpret_cast<node_id_t*>(location);
  }

  label_t* getNodeLabel(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_node_size_bytes);
    byte_offset += _data_size_bytes;
    byte_offset += (_M * sizeof(node_id_t));
    char* location = _index_memory + byte_offset;
    return reinterpret_cast<label_t*>(location);
  }

  inline void swapNodes(node_id_t a, node_id_t b, void* temp_data, node_id_t* temp_links,
                        label_t* temp_label) {

    // stash b in temp
    std::memcpy(temp_data, getNodeData(b), _data_size_bytes);
    std::memcpy(temp_links, getNodeLinks(b), _M * sizeof(node_id_t));
    std::memcpy(temp_label, getNodeLabel(b), sizeof(label_t));

    // place node at a in b
    std::memcpy(getNodeData(b), getNodeData(a), _data_size_bytes);
    std::memcpy(getNodeLinks(b), getNodeLinks(a), _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(b), getNodeLabel(a), sizeof(label_t));

    // put node b in a
    std::memcpy(getNodeData(a), temp_data, _data_size_bytes);
    std::memcpy(getNodeLinks(a), temp_links, _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(a), temp_label, sizeof(label_t));
  }

  /**
   * @brief Performs beam search for the nearest neighbors of the query.
   * @TODO: Add `entry_node_dist` argument to this function since we expect to
   * have computed that a priori.
   *
   * @param query               The query vector.
   * @param entry_node          The node to start the search from.
   * @param buffer_size         This is equivalent to `ef_search` in the HNSW
   *
   * @return PriorityQueue
   */

  PriorityQueue beamSearch(const void* query, const node_id_t entry_node, const int buffer_size) {
    PriorityQueue neighbors;
    PriorityQueue candidates;
    auto* visited_set = _visited_set_pool->pollAvailableSet();
    visited_set->clear();

    // Prefetch the data for entry node before computing its distance.
#ifdef USE_SSE
    _mm_prefetch(getNodeData(entry_node), _MM_HINT_T0);
#endif

    float dist = _distance->distance(/* x = */ query, /* y = */ getNodeData(entry_node),
                                     /* asymmetric = */ true);
    float max_dist = dist;
    candidates.emplace(-dist, entry_node);
    neighbors.emplace(dist, entry_node);
    visited_set->insert(entry_node);

    while (!candidates.empty()) {
      auto [distance, node] = candidates.top();

      if (-distance > max_dist && neighbors.size() >= buffer_size) {
        break;
      }

      candidates.pop();

      // Prefetching the next candidate node data and visited set marker
#ifdef USE_SSE
      if (!candidates.empty()) {
        _mm_prefetch(getNodeData(candidates.top().second), _MM_HINT_T0);
        visited_set->prefetch(candidates.top().second);
      }
#endif

      // TRACK EDGES BEFORE PROCESSING
      if (_track_edges) {
        trackEdgesForNode(node, visited_set);
      }

      processCandidateNode(
          /* query = */ query, /* node = */ node,
          /* max_dist = */ max_dist, /* buffer_size = */ buffer_size,
          /* visited_set = */ visited_set,
          /* neighbors = */ neighbors, /* candidates = */ candidates);
    }

    _visited_set_pool->pushVisitedSet(/* visited_set = */ visited_set);
    return neighbors;
  }

  void processCandidateNode(const void* query, node_id_t& node, float& max_dist, const int buffer_size,
                            VisitedSet* visited_set, PriorityQueue& neighbors, PriorityQueue& candidates) {
    // Lock all operations on this specific node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[node]);

    node_id_t* neighbor_node_links = getNodeLinks(node);
    for (uint32_t i = 0; i < _M; i++) {
      node_id_t neighbor_node_id = neighbor_node_links[i];

      // If using SSE, prefetch the next neighbor node data and the visited
      // marker
#ifdef USE_SSE
      if (i != _M - 1) {
        _mm_prefetch(getNodeData(neighbor_node_links[i + 1]), _MM_HINT_T0);
        visited_set->prefetch(neighbor_node_links[i + 1]);
      }
#endif

      bool neighbor_is_visited = visited_set->isVisited(/* num = */ neighbor_node_id);

      if (neighbor_is_visited) {
        continue;
      }
      visited_set->insert(/* num = */ neighbor_node_id);
      float dist = _distance->distance(/* x = */ query,
                                       /* y = */ getNodeData(neighbor_node_id),
                                       /* asymmetric = */ true);

      if (_collect_stats) {
        _distance_computations.fetch_add(1);
      }

      if (neighbors.size() < buffer_size || dist < max_dist) {
        candidates.emplace(-dist, neighbor_node_id);
        neighbors.emplace(dist, neighbor_node_id);
#ifdef USE_SSE
        _mm_prefetch(getNodeData(candidates.top().second), _MM_HINT_T0);
#endif
        if (neighbors.size() > buffer_size) {
          neighbors.pop();
        }
        if (!neighbors.empty()) {
          max_dist = neighbors.top().first;
        }
      }
    }
  }

  void selectNeighborsUnified(PriorityQueue& neighbors, int M, const void* query = nullptr,
                              node_id_t node_id = 0) {
    // Hub-aware selection takes precedence if enabled and hubs are identified
    if (_hub_aware_construction && _hubs_identified) {
      selectNeighborsHubAware(neighbors, M, query, node_id);
    } else if (_pruning_strategy == PruningStrategy::ALPHA_DIVERSITY) {
      selectNeighborsAlphaDiversity(neighbors, M, _alpha);
    } else if (_pruning_strategy == PruningStrategy::SSG) {
      if (query == nullptr) {
        throw std::runtime_error("SSG pruning requires query vector");
      }
      selectNeighborsSSG(neighbors, M, query);
    } else if (_pruning_strategy == PruningStrategy::RNG) {
      if (query == nullptr) {
        throw std::runtime_error("RNG pruning requires query vector");
      }
      selectNeighborsRNG(neighbors, M, query);
    } else {
      selectNeighbors(neighbors, M);
    }
  }

  /**
   * @brief Selects neighbors from the PriorityQueue, according to the HNSW
   * heuristic. The neighbors priority queue contains elements sorted by
   * distance where the top element is the furthest neighbor from the query.
   */
  void selectNeighborsAlphaDiversity(PriorityQueue& neighbors, int M, float alpha) {

    if (neighbors.size() <= M) {
      return;  // No pruning needed
    }

    // Convert max-heap to min-heap (sort by distance ascending)
    std::priority_queue<std::pair<float, node_id_t>, std::vector<std::pair<float, node_id_t>>,
                        std::greater<std::pair<float, node_id_t>>>
        candidates;

    while (!neighbors.empty()) {
      auto [dist, id] = neighbors.top();
      candidates.emplace(dist, id);
      neighbors.pop();
    }

    // Greedily select diverse neighbors
    std::vector<dist_node_t> selected;
    selected.reserve(M);

    while (!candidates.empty() && selected.size() < M) {
      auto [d_query, candidate] = candidates.top();
      candidates.pop();

      bool keep = true;
      for (const auto& [_, already_selected] : selected) {
        // Compute distance between candidate and already selected neighbor
        float d_neighbor = _distance->distance(
            /* x = */ getNodeData(candidate),
            /* y = */ getNodeData(already_selected));

        // Prune if too close to existing neighbor
        if (d_neighbor < d_query / (1.0f + alpha)) {
          keep = false;
          break;
        }
      }

      if (keep) {
        selected.push_back({d_query, candidate});
      }
    }

    // Put selected neighbors back into the priority queue
    for (const auto& [dist, id] : selected) {
      neighbors.emplace(dist, id);
    }
  }

  void selectNeighbors(PriorityQueue& neighbors, int M) {
    if (neighbors.size() < M) {
      return;
    }

    std::priority_queue<std::pair<float, node_id_t>> candidates;
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);

    while (neighbors.size() > 0) {
      auto [distance, id] = neighbors.top();

      candidates.emplace(-distance, id);
      neighbors.pop();
    }

    while (candidates.size() > 0) {
      if (saved_candidates.size() >= M) {
        break;
      }
      // Extract the closest element from candidates.
      auto [distance_to_query, current_node_id] = candidates.top();
      distance_to_query = -distance_to_query;
      candidates.pop();

      bool should_keep_candidate = true;
      for (const auto& [_, second_pair_node_id] : saved_candidates) {
        float cur_dist = _distance->distance(/* x = */ getNodeData(second_pair_node_id),
                                             /* y = */ getNodeData(current_node_id));

        if (cur_dist < distance_to_query) {
          should_keep_candidate = false;
          break;
        }
      }
      if (should_keep_candidate) {
        // We could do neighbors.emplace except we have to iterate
        // through saved_candidates, and std::priority_queue doesn't
        // support iteration (there is no technical reason why not).
        auto current_pair = std::make_pair(-distance_to_query, current_node_id);
        saved_candidates.push_back(current_pair);
      }
    }
    // TODO: implement my own priority queue, get rid of vector
    // saved_candidates, add directly to neighborqueue earlier.
    for (const dist_node_t& current_pair : saved_candidates) {
      neighbors.emplace(-current_pair.first, current_pair.second);
    }
  }

  void connectNeighbors(PriorityQueue& neighbors, node_id_t new_node_id) {
    // connects neighbors according to the HSNW heuristic

    // Lock all operations on this node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[new_node_id]);

    node_id_t* new_node_links = getNodeLinks(new_node_id);
    void* new_node_data = getNodeData(new_node_id);
    int i = 0;  // iterates through links for "new_node_id"

    while (neighbors.size() > 0) {
      node_id_t neighbor_node_id = neighbors.top().second;
      // add link to the current new node
      new_node_links[i] = neighbor_node_id;
      // now do the back-connections (a little tricky)

      std::unique_lock<std::mutex> neighbor_lock(_node_links_mutexes[neighbor_node_id]);
      node_id_t* neighbor_node_links = getNodeLinks(neighbor_node_id);
      void* neighbor_data = getNodeData(neighbor_node_id);
      bool is_inserted = false;
      for (size_t j = 0; j < _M; j++) {
        if (neighbor_node_links[j] == neighbor_node_id) {
          // If there is a self-loop, replace the self-loop with
          // the desired link.
          neighbor_node_links[j] = new_node_id;
          is_inserted = true;
          break;
        }
      }
      if (!is_inserted) {
        // now, we may to replace one of the links. This will disconnect
        // the old neighbor and create a directed edge, so we have to be
        // very careful. To ensure we respect the pruning heuristic, we
        // construct a candidate set including the old links AND our new
        // one, then prune this candidate set to get the new neighbors.

        float max_dist = _distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                             /* y = */ getNodeData(new_node_id));

        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (size_t j = 0; j < _M; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            auto label = neighbor_node_links[j];
            auto distance = _distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                                /* y = */ getNodeData(label));
            candidates.emplace(distance, label);
          }
        }
        // 2X larger than the previous call to selectNeighbors.
        selectNeighborsUnified(candidates, _M, neighbor_data, neighbor_node_id);
        // connect the pruned set of candidates, including self-loops:
        size_t j = 0;
        while (candidates.size() > 0) {  // candidates
          neighbor_node_links[j] = candidates.top().second;
          candidates.pop();
          j++;
        }
        while (j < _M) {  // self-loops (unused links)
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }

      // Unlock the current node we are iterating over
      neighbor_lock.unlock();

      // loop increments:
      i++;
      neighbors.pop();
    }
  }

  /**
   * @brief Selects a node to use as the entry point for a new node.
   * This proceeds in a greedy fashion, by selecting the node with
   * the smallest distance to the query.
   *
   * @param query
   * @param num_initializations
   * @return node_id_t
   */
  inline node_id_t initializeSearch(const void* query, int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument("num_initializations must be greater than 0.");
    }

    int step_size = _cur_num_nodes / num_initializations;
    step_size = step_size ? step_size : 1;

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    if (_collect_stats) {
      _distance_computations.fetch_add(num_initializations);
    }

    for (node_id_t node = 0; node < _cur_num_nodes; node += step_size) {
      float dist = _distance->distance(/* x = */ query, /* y = */ getNodeData(node),
                                       /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
    return entry_node;
  }

  void relabel(const std::vector<node_id_t>& P) {
    // 1. Rewire all of the node connections
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      node_id_t* links = getNodeLinks(n);
      for (int m = 0; m < _M; m++) {
        links[m] = P[links[m]];
      }
    }

    // 2. Physically re-layout the nodes (in place)
    char* temp_data = new char[_data_size_bytes];
    node_id_t* temp_links = new node_id_t[_M];
    label_t* temp_label = new label_t;

    auto* visited_set = _visited_set_pool->pollAvailableSet();

    // In this context, is_visited stores which nodes have been relocated
    // (it would be equivalent to name this variable "is_relocated").
    visited_set->clear();

    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      if (visited_set->isVisited(/* num = */ n)) {
        continue;
      }

      node_id_t src = n;
      node_id_t dest = P[src];

      // swap node at src with node at dest
      swapNodes(src, dest, temp_data, temp_links, temp_label);

      // mark src as having been relocated
      visited_set->insert(src);

      // recursively relocate the node from "dest"
      while (!visited_set->isVisited(/* num = */ dest)) {
        // mark node as having been relocated
        visited_set->insert(dest);
        // the value of src remains the same. However, dest needs
        // to change because the node located at src was previously
        // located at dest, and must be relocated to P[dest].
        dest = P[dest];

        // swap node at src with node at dest
        swapNodes(src, dest, temp_data, temp_links, temp_label);
      }
    }

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    delete[] temp_data;
    delete[] temp_links;
    delete temp_label;
  }
};

}  // namespace flatnav
