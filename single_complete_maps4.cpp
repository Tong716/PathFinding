#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <omp.h>

using namespace std;


// --- 1. 结构与维度定义 ---
// 对应数据集中的 4 个目标
enum CostType { DISTANCE, TRAVEL_TIME, ELEVATION, AVG_DEGREE };

struct Edge {
    int to;
    double distance;
    double travel_time;
    double elevation;
    double avg_degree;
};

struct PathCost {
    double dist = 0;
    double time = 0;
    double elev = 0;
    double deg = 0;
    double main_cost = 0; // 当前 Yen's 算法关注的主权重
    vector<int> path;

    bool is_same_path(const PathCost& other) const {
        return path == other.path;
    }
};

struct NodeDist {
    double f, h;
    int u;
    bool operator>(const NodeDist& other) const {
        if (abs(f - other.f) > 1e-9) return f > other.f;
        return h > other.h;
    }
};

vector<vector<Edge>> graph;
unordered_map<string, int> node_to_id;
vector<string> id_to_node;

// --- 2. 工具函数 ---

double get_weight(const Edge& e, CostType metric) {
    switch (metric) {
        case DISTANCE:    return e.distance;
        case TRAVEL_TIME: return e.travel_time;
        case ELEVATION:   return e.elevation;
        case AVG_DEGREE:  return e.avg_degree;
        default:          return e.distance;
    }
}

int get_node_id(const string &s) {
    auto it = node_to_id.find(s);
    if (it != node_to_id.end()) return it->second;
    int id = (int)node_to_id.size();
    node_to_id[s] = id;
    id_to_node.push_back(s);
    graph.push_back({});
    return id;
}

// 适配新格式: src dst distance travel_time elevation avg_degree
void load_graph(const string &filename) {
    ifstream fin(filename);
    if (!fin.is_open()) { cerr << "Error: Cannot open " << filename << endl; exit(1); }
    string line;
    while (getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue; // 跳过注释行
        stringstream ss(line);
        string u_str, v_str;
        double d, t, e, deg;
        if (!(ss >> u_str >> v_str >> d >> t >> e >> deg)) continue;
        
        int uid = get_node_id(u_str);
        int vid = get_node_id(v_str);
        graph[uid].push_back({vid, d, t, e, deg});
    }
    cout << "Graph loaded: " << graph.size() << " nodes" << endl;
}

PathCost calculate_metrics(const vector<int>& nodes, CostType current_metric) {
    PathCost pc; pc.path = nodes;
    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
        int u = nodes[i], v = nodes[i+1];
        for (auto& e : graph[u]) {
            if (e.to == v) {
                pc.dist += e.distance; pc.time += e.travel_time;
                pc.elev += e.elevation; pc.deg += e.avg_degree;
                pc.main_cost += get_weight(e, current_metric);
                break;
            }
        }
    }
    return pc;
}

// 4 维度非支配筛选
bool is_dominated(const PathCost &a, const PathCost &b) {
    bool d_le = a.dist <= b.dist + 1e-7;
    bool t_le = a.time <= b.time + 1e-7;
    bool e_le = a.elev <= b.elev + 1e-7;
    bool g_le = a.deg <= b.deg + 1e-7;
    
    bool strictly_better = (a.dist < b.dist - 1e-7) || (a.time < b.time - 1e-7) || 
                           (a.elev < b.elev - 1e-7) || (a.deg < b.deg - 1e-7);
    
    return (d_le && t_le && e_le && g_le) && strictly_better;
}

// --- 3. 核心算法 (A* & Yen's) ---

pair<double, vector<int>> a_star(int source, int target, CostType metric, 
                                 const vector<double>& h,
                                 const set<int>& f_nodes,
                                 const set<pair<int, int>>& f_edges) {
    int n = (int)graph.size();
    vector<double> g(n, numeric_limits<double>::infinity());
    vector<int> parent(n, -1);
    priority_queue<NodeDist, vector<NodeDist>, greater<NodeDist>> pq;

    if (f_nodes.count(source)) return {-1, {}};
    g[source] = 0;
    pq.push({h[source], h[source], source});

    while (!pq.empty()) {
        NodeDist top = pq.top(); pq.pop();
        int u = top.u;
        if (u == target) break;
        if (top.f > g[u] + h[u] + 1e-9) continue;

        for (auto& e : graph[u]) {
            if (f_nodes.count(e.to) || f_edges.count({u, e.to})) continue;
            double weight = get_weight(e, metric);
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

vector<double> precompute_h(int target, CostType metric) {
    int n = (int)graph.size();
    vector<double> h(n, numeric_limits<double>::infinity());
    vector<vector<pair<int, double>>> rev(n);
    for(int u=0; u<n; ++u)
        for(auto &e : graph[u]) rev[e.to].push_back({u, get_weight(e, metric)});
    
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
    h[target] = 0; pq.push({0, target});
    while(!pq.empty()){
        double d = pq.top().first; int u = pq.top().second; pq.pop();
        if(d > h[u]) continue;
        for(auto &edge : rev[u]){
            if(h[u] + edge.second < h[edge.first]){
                h[edge.first] = h[u] + edge.second;
                pq.push({h[edge.first], edge.first});
            }
        }
    }
    return h;
}

vector<PathCost> find_k_shortest(int s, int t, int K, CostType metric) {
    vector<PathCost> A;
    auto h_func = precompute_h(t, metric);
    auto first = a_star(s, t, metric, h_func, {}, {});
    if (first.first < 0) return A;

    auto comp = [](const pair<double, vector<int>>& p1, const pair<double, vector<int>>& p2) {
        if (abs(p1.first - p2.first) > 1e-9) return p1.first > p2.first;
        return p1.second > p2.second; 
    };
    priority_queue<pair<double, vector<int>>, vector<pair<double, vector<int>>>, decltype(comp)> B(comp);
    B.push(first);
    set<vector<int>> seen; seen.insert(first.second);

    // 用于记录出现了多少种不同的成本
    set<double> distinct_costs;
    double max_allowed_cost = numeric_limits<double>::infinity();
    // --- 关键修改点：增加一个硬性的最大路径数量限制 ---
    const int MAX_TOTAL_PATHS = 100; // 每个维度最多捞 100 条，防止死循环

    while (!B.empty()) {
        auto top = B.top(); B.pop();
        double current_cost = top.first;

        // 如果当前成本已经超过了我们允许的第 K 档成本，直接收工
        if (current_cost > max_allowed_cost + 1e-7) break;
        // 逻辑 B：虽然还在第 K 档内，但数量实在太多了，也强制停
        if (A.size() >= MAX_TOTAL_PATHS) break;
        
        // 记录这条路径
        A.push_back(calculate_metrics(top.second, metric));
        
        // 更新成本档位
        distinct_costs.insert(current_cost);
        
        // 如果我们刚刚凑齐了第 K 种不同的成本，设置阈值
        if (distinct_costs.size() == (size_t)K) {
            max_allowed_cost = current_cost;
        }

        // --- Yen's 算法的节点偏移逻辑保持不变 ---
        vector<int> last_path = top.second;
        for (size_t i = 0; i < last_path.size() - 1; ++i) {
            int spurNode = last_path[i];
            vector<int> rootPath(last_path.begin(), last_path.begin() + i + 1);
            set<pair<int, int>> f_edges;
            // 注意：这里需要遍历 A 中所有已找到的路径来排除边
            for (auto& p_obj : A) {
                if (p_obj.path.size() > i + 1 && 
                    equal(rootPath.begin(), rootPath.end(), p_obj.path.begin())) {
                    f_edges.insert({p_obj.path[i], p_obj.path[i+1]});
                }
            }
            set<int> f_nodes(last_path.begin(), last_path.begin() + i);

            auto spur = a_star(spurNode, t, metric, h_func, f_nodes, f_edges);
            if (spur.first >= 0) {
                vector<int> total = rootPath;
                total.insert(total.end(), spur.second.begin() + 1, spur.second.end());
                if (seen.find(total) == seen.end()) {
                    B.push({calculate_metrics(total, metric).main_cost, total});
                    seen.insert(total);
                }
            }
        }
    }
    return A;
}

// --- 4. 执行入口 ---

int main() {
    // 这里的地名需要根据你实际的文件名修改，例如 edge_BAY.txt
    string city = "FLA";
    load_graph("../dataset/DIMACS/edges_" + city + ".txt");
    
    ifstream fin_pairs("../dataset/DIMACS/point_pairs_numeric.csv");
    string header; getline(fin_pairs, header);
    vector<pair<string, string>> tasks;
    string line;
    int max_tasks = 100; 
    while (getline(fin_pairs, line) && tasks.size() < max_tasks) {
        stringstream ss(line); string s, t;
        if (getline(ss, s, ',') && getline(ss, t, ',')) {
            tasks.push_back({s, t});
        }
    }
    fin_pairs.close();    
    
    // while (getline(fin_pairs, line)) {
    //     stringstream ss(line); string s, t;
    //     if (getline(ss, s, ',') && getline(ss, t, ',')) tasks.push_back({s, t});
    // }

    ofstream fout("../dataset/DIMACS/result/merged_" + city + "_4obj.csv");
    fout << "src,dest,distance,travel_time,elevation,avg_degree,path\n";

    int K = 3; 
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)tasks.size(); ++i) {
        string s_str = tasks[i].first;
        string t_str = tasks[i].second;
        if (node_to_id.count(s_str) == 0 || node_to_id.count(t_str) == 0) continue;
        
        int s_id = node_to_id[s_str];
        int t_id = node_to_id[t_str];

        // 1. 跑 4 个维度的 K 短路
        vector<PathCost> all_candidates;
        CostType metrics[] = {DISTANCE, TRAVEL_TIME, ELEVATION, AVG_DEGREE};
        for (auto m : metrics) {
            auto res = find_k_shortest(s_id, t_id, K, m);
            all_candidates.insert(all_candidates.end(), res.begin(), res.end());
        }

        // 2. 内存内非支配过滤 + 去重
        vector<PathCost> pareto_front;
        for (size_t j = 0; j < all_candidates.size(); ++j) {
            bool dominated = false;
            for (size_t k = 0; k < all_candidates.size(); ++k) {
                if (j == k) continue;
                if (is_dominated(all_candidates[k], all_candidates[j])) {
                    dominated = true; break;
                }
            }
            if (!dominated) {
                bool exists = false;
                for (auto &pf : pareto_front) {
                    if (pf.is_same_path(all_candidates[j])) { exists = true; break; }
                }
                if (!exists) pareto_front.push_back(all_candidates[j]);
            }
        }

        // 3. 结果汇总
        stringstream ss_out;
        for (auto &p : pareto_front) {
            ss_out << s_str << "," << t_str << "," << p.dist << "," << p.time << "," 
                   << p.elev << "," << p.deg << ",\"";
            for (size_t n_idx = 0; n_idx < p.path.size(); ++n_idx)
                ss_out << id_to_node[p.path[n_idx]] << (n_idx + 1 < p.path.size() ? "->" : "");
            ss_out << "\"\n";
        }

        #pragma omp critical
        {
            fout << ss_out.str();
            cout << "[" << city << "] Task " << i+1 << "/" << tasks.size() 
                 << " (" << s_str << "->" << t_str << ") Pareto Size: " << pareto_front.size() << endl;
        }
    }

    fout.close();
    return 0;
}
