#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <flatnav/index/Index.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/util/Datatype.h>

using namespace flatnav;
using namespace flatnav::distances;
using flatnav::util::DataType;

struct Point2D {
    float x, y;
    int id;
    std::string label;
};

void printPoint(const Point2D& p) {
    std::cout << "Point " << p.id << " (" << p.label << "): [" 
              << std::fixed << std::setprecision(2) 
              << p.x << ", " << p.y << "]";
}

void visualize2DSpace(const std::vector<Point2D>& points, 
                      const Point2D* query = nullptr,
                      const std::vector<int>* neighbors = nullptr) {
    const int gridSize = 40;
    const int gridHeight = 20;
    std::vector<std::vector<char>> grid(gridHeight, std::vector<char>(gridSize, '.'));
    
    float minX = -10, maxX = 10;
    float minY = -10, maxY = 10;
    
    for (const auto& p : points) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }
    
    auto toGridX = [&](float x) {
        return static_cast<int>((x - minX) / (maxX - minX) * (gridSize - 1));
    };
    
    auto toGridY = [&](float y) {
        return static_cast<int>((1.0f - (y - minY) / (maxY - minY)) * (gridHeight - 1));
    };
    
    for (const auto& p : points) {
        int gx = toGridX(p.x);
        int gy = toGridY(p.y);
        if (gx >= 0 && gx < gridSize && gy >= 0 && gy < gridHeight) {
            bool isNeighbor = false;
            if (neighbors) {
                for (int n : *neighbors) {
                    if (n == p.id) {
                        isNeighbor = true;
                        break;
                    }
                }
            }
            
            if (isNeighbor) {
                grid[gy][gx] = '*';
            } else if (p.id < 10) {
                grid[gy][gx] = '0' + p.id;
            } else {
                grid[gy][gx] = 'o';
            }
        }
    }
    
    if (query) {
        int gx = toGridX(query->x);
        int gy = toGridY(query->y);
        if (gx >= 0 && gx < gridSize && gy >= 0 && gy < gridHeight) {
            grid[gy][gx] = 'Q';
        }
    }
    
    std::cout << "\n2D Space Visualization:\n";
    std::cout << "  ";
    for (int i = 0; i < gridSize; i += 5) {
        std::cout << "|";
        for (int j = 1; j < 5 && i + j < gridSize; j++) {
            std::cout << " ";
        }
    }
    std::cout << "\n";
    
    for (int y = 0; y < gridHeight; y++) {
        if (y % 5 == 0) std::cout << "- ";
        else std::cout << "  ";
        for (int x = 0; x < gridSize; x++) {
            std::cout << grid[y][x];
        }
        std::cout << "\n";
    }
    
    std::cout << "  ";
    for (int i = 0; i < gridSize; i += 5) {
        std::cout << "|";
        for (int j = 1; j < 5 && i + j < gridSize; j++) {
            std::cout << " ";
        }
    }
    std::cout << "\n";
    
    std::cout << "\nLegend: Q=Query, *=Neighbor, o/0-9=Data points, .=empty\n";
}

std::vector<Point2D> createTestDataset() {
    std::vector<Point2D> points;
    
    points.push_back({-8.0f, 8.0f, 0, "top-left"});
    points.push_back({8.0f, 8.0f, 1, "top-right"});
    points.push_back({-8.0f, -8.0f, 2, "bottom-left"});
    points.push_back({8.0f, -8.0f, 3, "bottom-right"});
    points.push_back({0.0f, 0.0f, 4, "center"});
    
    points.push_back({-4.0f, 4.0f, 5, "mid-top-left"});
    points.push_back({4.0f, 4.0f, 6, "mid-top-right"});
    points.push_back({-4.0f, -4.0f, 7, "mid-bottom-left"});
    points.push_back({4.0f, -4.0f, 8, "mid-bottom-right"});
    
    points.push_back({0.0f, 6.0f, 9, "top-center"});
    points.push_back({0.0f, -6.0f, 10, "bottom-center"});
    points.push_back({-6.0f, 0.0f, 11, "left-center"});
    points.push_back({6.0f, 0.0f, 12, "right-center"});
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    for (int i = 13; i < 20; i++) {
        points.push_back({dist(rng), dist(rng), i, "random-" + std::to_string(i)});
    }
    
    return points;
}

int main() {
    std::cout << "=== FlatNav 2D Nearest Neighbor Search Demo ===\n\n";
    
    auto points = createTestDataset();
    const size_t dataset_size = points.size();
    // Use dimension 4 to avoid SIMD issues, we'll pad with zeros
    const size_t dimension = 4;
    
    std::cout << "Dataset contains " << dataset_size << " points in 2D space\n";
    std::cout << "(Using " << dimension << "D vectors internally for SIMD compatibility)\n\n";
    
    for (const auto& p : points) {
        printPoint(p);
        std::cout << "\n";
    }
    
    visualize2DSpace(points);
    
    std::cout << "\n--- Building FlatNav Index ---\n";
    
    const size_t max_edges_per_node = 6;
    const size_t ef_construction = 20;
    
    std::cout << "Creating distance function for dimension: " << dimension << "\n";
    auto distance = SquaredL2Distance<DataType::float32>::create(dimension);
    if (!distance) {
        std::cerr << "Failed to create distance function\n";
        return 1;
    }
    
    std::cout << "Creating index with dataset_size: " << dataset_size 
              << ", max_edges: " << max_edges_per_node << "\n";
    auto* index = new Index<SquaredL2Distance<DataType::float32>, int>(
        std::move(distance), 
        dataset_size,
        max_edges_per_node
    );
    
    std::cout << "Setting threads to 1\n";
    index->setNumThreads(1);
    
    std::cout << "Preparing data for indexing (padding 2D points to 4D)\n";
    std::vector<float> flat_data;
    flat_data.reserve(dataset_size * dimension);
    for (const auto& p : points) {
        flat_data.push_back(p.x);
        flat_data.push_back(p.y);
        flat_data.push_back(0.0f);  // padding
        flat_data.push_back(0.0f);  // padding
    }
    
    std::vector<int> labels;
    for (const auto& p : points) {
        labels.push_back(p.id);
    }
    
    std::cout << "Data size: " << flat_data.size() << " floats\n";
    std::cout << "Labels size: " << labels.size() << "\n";
    std::cout << "Calling addBatch with ef_construction: " << ef_construction << "\n";
    
    index->template addBatch<float>(
        (void*)flat_data.data(),
        labels,
        ef_construction
    );
    
    std::cout << "addBatch completed successfully\n";
    
    std::cout << "Index built successfully with:\n";
    std::cout << "  - Max edges per node: " << max_edges_per_node << "\n";
    std::cout << "  - EF construction: " << ef_construction << "\n\n";
    
    std::cout << "--- Performing Searches ---\n\n";
    
    std::vector<Point2D> queries = {
        {1.0f, 1.0f, -1, "Query1"},
        {-7.0f, 7.0f, -2, "Query2"},
        {5.0f, -3.0f, -3, "Query3"}
    };
    
    const size_t k = 5;
    const size_t ef_search = 20;
    
    for (const auto& q : queries) {
        std::cout << "========================================\n";
        std::cout << "Searching for " << k << " nearest neighbors to ";
        printPoint(q);
        std::cout << "\n";
        
        float query_data[4] = {q.x, q.y, 0.0f, 0.0f};  // Pad to 4D
        auto results = index->search(query_data, k, ef_search);
        
        std::cout << "\nResults (K=" << k << ", ef_search=" << ef_search << "):\n";
        std::cout << "Rank | Distance | Point Details\n";
        std::cout << "-----|----------|---------------\n";
        
        std::vector<int> neighbor_ids;
        for (size_t i = 0; i < results.size(); i++) {
            float dist = results[i].first;
            int label = results[i].second;
            neighbor_ids.push_back(label);
            
            const auto& found_point = points[label];
            float euclidean = std::sqrt(dist);
            
            std::cout << std::setw(4) << (i + 1) << " | " 
                     << std::fixed << std::setprecision(4) 
                     << std::setw(8) << euclidean << " | ";
            printPoint(found_point);
            std::cout << "\n";
        }
        
        visualize2DSpace(points, &q, &neighbor_ids);
        std::cout << "\n";
    }
    
    std::cout << "--- Understanding FlatNav ---\n\n";
    std::cout << "FlatNav is a graph-based index that:\n";
    std::cout << "1. Creates a navigable small world graph WITHOUT hierarchy\n";
    std::cout << "2. Each point connects to up to " << max_edges_per_node << " neighbors\n";
    std::cout << "3. During search, it navigates the graph greedily\n";
    std::cout << "4. Uses ef_search parameter to control search quality/speed tradeoff\n";
    std::cout << "5. In high dimensions, natural 'hubs' form that aid navigation\n\n";
    
    std::cout << "Key differences from HNSW:\n";
    std::cout << "- No hierarchical layers (flat structure)\n";
    std::cout << "- ~38% less memory usage\n";
    std::cout << "- Simpler implementation\n";
    std::cout << "- Same performance for high-dimensional data (d > 32)\n\n";
    
    // Note: Index destructor will handle cleanup
    delete index;
    
    return 0;
}