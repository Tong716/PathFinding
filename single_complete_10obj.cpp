#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <queue>
#include <set>
#include <unordered_set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

struct Edge {
    int to;
    vector<double> weights;
};

struct PathCost {
    vector<double> costs;
    double main_cost = 0.0;
    vector<int> path;

    bool is_same_path(const PathCost &other) const {
        return path == other.path;
    }
};

struct NodeDist {
    double f, h;
    int u;
    bool operator>(const NodeDist &other) const {
        if (abs(f - other.f) > 1e-9) return f > other.f;
        return h > other.h;
    }
};

vector<vector<Edge>> graph;
int num_objectives = 0;
int num_nodes = 0;

inline long long edge_key(int u, int v) {
    return (static_cast<long long>(u) << 32) | static_cast<unsigned long long>(v);
}

bool load_graph_directory(const string &dir_path, int expected_objectives) {
    vector<string> gr_files;
    for (auto const &entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;
        auto p = entry.path();
        if (p.extension() == ".gr") gr_files.push_back(p.string());
    }
    if (gr_files.empty()) {
        cerr << "Error: no .gr files found in " << dir_path << endl;
        return false;
    }
    sort(gr_files.begin(), gr_files.end());
    if (expected_objectives > 0 && (int)gr_files.size() < expected_objectives) {
        cerr << "Error: found " << gr_files.size() << " .gr files but expected " << expected_objectives << endl;
        return false;
    }
    if (expected_objectives <= 0) expected_objectives = (int)gr_files.size();
    gr_files.resize(expected_objectives);

    num_objectives = expected_objectives;
    graph.clear();
    num_nodes = 0;

    unordered_map<long long, int> edge_index;
    for (int obj = 0; obj < num_objectives; ++obj) {
        const string &filename = gr_files[obj];
        ifstream fin(filename);
        if (!fin.is_open()) {
            cerr << "Error: cannot open " << filename << endl;
            return false;
        }
        string line;
        int file_nodes = 0;
        int file_edges = 0;
        int added_edges = 0;
        while (getline(fin, line)) {
            if (line.empty()) continue;
            if (line[0] == 'c' || line[0] == 'p') {
                if (line[0] == 'p') {
                    string token;
                    stringstream ss(line);
                    ss >> token >> token >> file_nodes >> file_edges;
                    num_nodes = max(num_nodes, file_nodes);
                }
                continue;
            }
            if (line[0] != 'a') continue;
            stringstream ss(line);
            char type;
            int u, v;
            double w;
            ss >> type >> u >> v >> w;
            if (u <= 0 || v <= 0) continue;
            num_nodes = max(num_nodes, max(u, v));
            if ((int)graph.size() <= num_nodes) graph.resize(num_nodes + 1);

            long long key = edge_key(u, v);
            if (obj == 0) {
                Edge edge;
                edge.to = v;
                edge.weights.assign(num_objectives, 0.0);
                edge.weights[obj] = w;
                graph[u].push_back(move(edge));
                edge_index[key] = (int)graph[u].size() - 1;
                ++added_edges;
            } else {
                auto it = edge_index.find(key);
                if (it != edge_index.end()) {
                    graph[u][it->second].weights[obj] = w;
                } else {
                    Edge edge;
                    edge.to = v;
                    edge.weights.assign(num_objectives, 0.0);
                    edge.weights[obj] = w;
                    graph[u].push_back(move(edge));
                    edge_index[key] = (int)graph[u].size() - 1;
                    ++added_edges;
                }
            }
        }
        fin.close();
        cout << "Loaded objective " << obj + 1 << " from " << filename << ": nodes=" << num_nodes
             << ", edges=" << added_edges << endl;
    }
    return true;
}

vector<int> parse_query_line(const string &line) {
    stringstream ss(line);
    int s, t;
    if (!(ss >> s >> t)) return {};
    return {s, t};
}

vector<pair<int,int>> load_queries(const string &query_file) {
    vector<pair<int,int>> queries;
    ifstream fin(query_file);
    if (!fin.is_open()) {
        cerr << "Warning: cannot open query file " << query_file << "; using default pair\n";
        return queries;
    }
    string line;
    // Expect CSV with header like: query_id,source,target,eps
    bool first = true;
    unordered_set<long long> seen;
    const size_t MAX_QUERIES = 100;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            if (line.find("source") != string::npos && line.find("target") != string::npos) continue;
        }
        // split by comma
        stringstream ss(line);
        string item;
        vector<string> cols;
        while (getline(ss, item, ',')) {
            size_t a = item.find_first_not_of(" \t\r\n");
            size_t b = item.find_last_not_of(" \t\r\n");
            if (a == string::npos) cols.push_back(""); else cols.push_back(item.substr(a, b - a + 1));
        }
        if (cols.size() >= 3) {
            try {
                int s = stoi(cols[1]);
                int t = stoi(cols[2]);
                long long key = (static_cast<long long>(s) << 32) | static_cast<unsigned long long>(t);
                if (seen.insert(key).second) {
                    queries.emplace_back(s, t);
                }
            } catch (...) {
                // ignore parse errors
            }
        }
        if (queries.size() >= MAX_QUERIES) break; // only keep first 100 unique pairs
    }
    return queries;
}

PathCost calculate_metrics(const vector<int> &path, int metric) {
    PathCost pc;
    pc.costs.assign(num_objectives, 0.0);
    pc.path = path;
    pc.main_cost = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        int u = path[i];
        int v = path[i + 1];
        bool found = false;
        for (const auto &e : graph[u]) {
            if (e.to == v) {
                for (int o = 0; o < num_objectives; ++o) {
                    pc.costs[o] += e.weights[o];
                }
                pc.main_cost += e.weights[metric];
                found = true;
                break;
            }
        }
        if (!found) {
            cerr << "Warning: edge " << u << "->" << v << " not found in graph\n";
        }
    }
    return pc;
}

bool is_dominated(const PathCost &a, const PathCost &b) {
    bool all_le = true;
    bool strictly_better = false;
    const double eps = 1e-7;
    for (int i = 0; i < num_objectives; ++i) {
        if (a.costs[i] > b.costs[i] + eps) {
            all_le = false;
            break;
        }
        if (a.costs[i] < b.costs[i] - eps) {
            strictly_better = true;
        }
    }
    return all_le && strictly_better;
}

pair<double, vector<int>> a_star(int source, int target, int metric,
                                 const vector<double> &h,
                                 const set<int> &forbidden_nodes,
                                 const set<pair<int,int>> &forbidden_edges) {
    int n = num_nodes;
    vector<double> g(n + 1, numeric_limits<double>::infinity());
    vector<int> parent(n + 1, -1);
    priority_queue<NodeDist, vector<NodeDist>, greater<NodeDist>> pq;

    if (forbidden_nodes.count(source)) return {-1, {}};
    g[source] = 0.0;
    pq.push({h[source], h[source], source});

    while (!pq.empty()) {
        auto top = pq.top();
        pq.pop();
        int u = top.u;
        if (u == target) break;
        if (top.f > g[u] + h[u] + 1e-9) continue;
        for (const auto &e : graph[u]) {
            if (forbidden_nodes.count(e.to)) continue;
            if (forbidden_edges.count({u, e.to})) continue;
            double weight = e.weights[metric];
            if (g[u] + weight < g[e.to]) {
                g[e.to] = g[u] + weight;
                parent[e.to] = u;
                pq.push({g[e.to] + h[e.to], h[e.to], e.to});
            }
        }
    }
    if (g[target] == numeric_limits<double>::infinity()) return {-1, {}};
    vector<int> path;
    for (int v = target; v != -1; v = parent[v]) path.push_back(v);
    reverse(path.begin(), path.end());
    return {g[target], path};
}

vector<double> precompute_h(int target, int metric) {
    int n = num_nodes;
    vector<double> h(n + 1, numeric_limits<double>::infinity());
    vector<vector<pair<int,double>>> rev(n + 1);
    for (int u = 1; u <= n; ++u) {
        for (const auto &e : graph[u]) {
            rev[e.to].push_back({u, e.weights[metric]});
        }
    }
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<pair<double,int>>> pq;
    h[target] = 0.0;
    pq.push({0.0, target});
    while (!pq.empty()) {
        auto [dist, u] = pq.top();
        pq.pop();
        if (dist > h[u]) continue;
        for (const auto &pr : rev[u]) {
            int v = pr.first;
            double w = pr.second;
            if (h[u] + w < h[v]) {
                h[v] = h[u] + w;
                pq.push({h[v], v});
            }
        }
    }
    return h;
}

vector<PathCost> find_k_shortest(int s, int t, int K, int metric) {
    vector<PathCost> result;
    auto h_func = precompute_h(t, metric);
    auto first = a_star(s, t, metric, h_func, {}, {});
    if (first.first < 0) return result;

    auto comp = [](const pair<double, vector<int>> &a,
                   const pair<double, vector<int>> &b) {
        if (abs(a.first - b.first) > 1e-9) return a.first > b.first;
        return a.second > b.second;
    };

    priority_queue<pair<double, vector<int>>, vector<pair<double, vector<int>>>, decltype(comp)> B(comp);
    set<vector<int>> seen;
    B.push(first);
    seen.insert(first.second);

    set<double> distinct_costs;
    double max_allowed_cost = numeric_limits<double>::infinity();
    const int MAX_TOTAL_PATHS = 100;

    while (!B.empty()) {
        auto [current_cost, current_path] = B.top();
        B.pop();
        if (current_cost > max_allowed_cost + 1e-7) break;
        if ((int)result.size() >= MAX_TOTAL_PATHS) break;

        result.push_back(calculate_metrics(current_path, metric));
        distinct_costs.insert(current_cost);
        if ((int)distinct_costs.size() == K) {
            max_allowed_cost = current_cost;
        }

        for (size_t i = 0; i + 1 < current_path.size(); ++i) {
            int spur_node = current_path[i];
            vector<int> root_path(current_path.begin(), current_path.begin() + i + 1);
            set<pair<int,int>> forbidden_edges;
            for (const auto &pc : result) {
                if (pc.path.size() > i + 1 && equal(root_path.begin(), root_path.end(), pc.path.begin())) {
                    forbidden_edges.insert({pc.path[i], pc.path[i + 1]});
                }
            }
            set<int> forbidden_nodes(current_path.begin(), current_path.begin() + i);
            auto spur = a_star(spur_node, t, metric, h_func, forbidden_nodes, forbidden_edges);
            if (spur.first >= 0) {
                vector<int> total_path = root_path;
                total_path.insert(total_path.end(), spur.second.begin() + 1, spur.second.end());
                if (seen.insert(total_path).second) {
                    double total_cost = calculate_metrics(total_path, metric).main_cost;
                    B.push({total_cost, move(total_path)});
                }
            }
        }
    }
    return result;
}

void print_usage() {
    cout << "Usage: single_complete_10obj --dir <graph_dir> [options]\n";
    cout << "  --dir <graph_dir>         directory containing 10 .gr objective files\n";
    cout << "  --objs <n>               number of objective files to read (default 10)\n";
    cout << "  --source <node>          source node id (1-based, default 1)\n";
    cout << "  --target <node>          target node id (1-based, default last node)\n";
    cout << "  --query-file <file>      optional query file with src dst per line\n";
    cout << "  --K <k>                  K shortest paths for each objective (default 3)\n";
    cout << "  --output <file>          output CSV file (default results_10obj.csv)\n";
}

int main(int argc, char *argv[]) {
    string graph_dir;
    string query_file = "/data/home/XieTong/route_planning/Multi-Objective-Search-Benchmarks-main/benchmarks/road/ger10/queries/ger10_10000/ger10_connected_10000_queries.csv";
    string output_file = "/data/home/XieTong/route_planning/PSO_HP/dataset/ger/results_10obj.csv";
    int source = 1;
    int target = -1;
    int K = 3;
    int expected_objs = 10;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dir" && i + 1 < argc) {
            graph_dir = argv[++i];
        } else if (arg == "--query-file" && i + 1 < argc) {
            query_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            source = stoi(argv[++i]);
        } else if (arg == "--target" && i + 1 < argc) {
            target = stoi(argv[++i]);
        } else if (arg == "--K" && i + 1 < argc) {
            K = stoi(argv[++i]);
        } else if (arg == "--objs" && i + 1 < argc) {
            expected_objs = stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else {
            cerr << "Unknown argument: " << arg << endl;
            print_usage();
            return 1;
        }
    }

    if (graph_dir.empty()) {
        cerr << "Error: --dir is required\n";
        print_usage();
        return 1;
    }

    if (!load_graph_directory(graph_dir, expected_objs)) {
        return 1;
    }
    if (target < 0) target = num_nodes;
    if (source < 1 || source > num_nodes || target < 1 || target > num_nodes) {
        cerr << "Error: source/target out of range [1," << num_nodes << "]\n";
        return 1;
    }

    vector<pair<int,int>> queries;
    if (!query_file.empty()) {
        queries = load_queries(query_file);
        if (queries.empty()) {
            cerr << "Warning: no queries loaded from " << query_file << "; using default pair\n";
        }
    }
    if (queries.empty()) {
        queries.emplace_back(source, target);
    }

    fs::path output_path(output_file);
    if (!output_path.has_parent_path()) {
        cerr << "Error: invalid output file path " << output_file << endl;
        return 1;
    }
    fs::create_directories(output_path.parent_path());
    ofstream fout(output_file);
    if (!fout.is_open()) {
        cerr << "Error: cannot open output file " << output_file << endl;
        return 1;
    }

    // 输出 header: src,dst,cost1..costN,runtime,path
    fout << "src,dst";
    for (int o = 0; o < num_objectives; ++o) {
        fout << ",cost" << o + 1;
    }
    fout << ",runtime,path\n";

    for (size_t qi = 0; qi < queries.size(); ++qi) {
        int s = queries[qi].first;
        int t = queries[qi].second;
        if (s < 1 || s > num_nodes || t < 1 || t > num_nodes) {
            cerr << "Skipping invalid query " << s << " " << t << "\n";
            continue;
        }
        cout << "Processing query " << qi + 1 << "/" << queries.size() << ": " << s << "->" << t << "\n";
        auto t0 = chrono::steady_clock::now();
        vector<PathCost> candidates;
        #pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < num_objectives; ++m) {
            auto metric_paths = find_k_shortest(s, t, K, m);
            #pragma omp critical
            {
                candidates.insert(candidates.end(), metric_paths.begin(), metric_paths.end());
            }
        }

        vector<PathCost> pareto_front;
        for (size_t i = 0; i < candidates.size(); ++i) {
            bool dominated = false;
            for (size_t j = 0; j < candidates.size(); ++j) {
                if (i == j) continue;
                if (is_dominated(candidates[j], candidates[i])) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                bool exists = false;
                for (const auto &pc : pareto_front) {
                    if (pc.is_same_path(candidates[i])) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) pareto_front.push_back(candidates[i]);
            }
        }
        auto t1 = chrono::steady_clock::now();
        double runtime = chrono::duration_cast<chrono::duration<double>>(t1 - t0).count();

        cout << "Found " << pareto_front.size() << " nondominated paths for query " << s << "->" << t << "\n";
        for (const auto &pc : pareto_front) {
            fout << s << "," << t;
            for (double cost : pc.costs) {
                fout << "," << fixed << setprecision(6) << cost;
            }
            fout << "," << fixed << setprecision(6) << runtime;
            fout << ",\"";
            for (size_t ii = 0; ii < pc.path.size(); ++ii) {
                fout << pc.path[ii];
                if (ii + 1 < pc.path.size()) fout << "->";
            }
            fout << "\"\n";
        }
    }

    fout.close();
    cout << "Written results to " << output_file << "\n";
    return 0;
}
