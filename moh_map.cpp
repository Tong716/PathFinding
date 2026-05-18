#include <climits>
#include <cstddef>
#include <cctype>
#include <iostream>
#include <memory>
#include <time.h>
#include <fstream>

#include "AbstractSolver.h"
#include "ApexSearch.h"
#include "AnytimeApexSearch.h"
#include "WCApexSearch.h"
#include "BOAStar.h"
#include "AnytimeBOA.h"
#include "GCL_bucket.h"
#include "ShortestPathHeuristic.h"
#include "LTMOA.h"
#include "LTMOA2.h"
#include "Utils/Definitions.h"
#include "Utils/IOUtils.h"

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>

using namespace std;
namespace po = boost::program_options;

const std::string resource_path = "resources/";
// const std::string output_path = "output/";
const std::string output_path = "";
using namespace std::chrono_literals;

std::string dir_loc_sol;
bool log_sol;

bool load_edge_txt_file(std::string txt_file, std::vector<Edge> &edges_out, size_t &graph_size) {
    std::ifstream file(txt_file.c_str());
    if (file.is_open() == false) {
        std::cerr << "cannot open edge file " << txt_file << std::endl;
        return false;
    }

    std::string line;
    size_t max_node_num = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::vector<std::string> decomposed_line;
        split_string(line, " \t", decomposed_line);
        if (decomposed_line.size() < 4) {
            continue;
        }

        auto source = std::stoul(decomposed_line[0]);
        auto target = std::stoul(decomposed_line[1]);
        std::vector<cost_t> costs;
        for (size_t i = 2; i < decomposed_line.size(); ++i) {
            costs.push_back((cost_t)std::stoul(decomposed_line[i]));
        }

        edges_out.emplace_back((cost_t)source, (cost_t)target, costs);
        max_node_num = std::max({max_node_num, source, target});
    }
    graph_size = max_node_num + 1;
    return true;
}

bool load_query_csv(std::string query_file, std::vector<std::pair<size_t, size_t>> &queries_out) {
    std::ifstream file(query_file.c_str());
    if (file.is_open() == false) {
        return false;
    }

    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::vector<std::string> decomposed_line;
        split_string(line, ",", decomposed_line);
        if (decomposed_line.size() < 2) {
            continue;
        }

        if (first_line) {
            std::string token0 = decomposed_line[0];
            for (auto &c: token0) c = std::tolower(c);
            if (token0 == "source" || token0 == "src" || token0 == "s") {
                first_line = false;
                continue;
            }
            first_line = false;
        }

        queries_out.emplace_back(std::stoul(decomposed_line[0]), std::stoul(decomposed_line[1]));
    }
    return true;
}

bool load_query_pairs(std::string query_file, std::vector<std::pair<size_t, size_t>> &queries_out) {
    if (query_file.size() >= 4 && query_file.substr(query_file.size() - 4) == ".csv") {
        return load_query_csv(query_file, queries_out);
    }
    return load_queries(query_file, queries_out);
}

void single_run_map(AdjacencyMatrix& graph, AdjacencyMatrix&inv_graph, size_t source, size_t target, std::ofstream& output, std::string algorithm , double eps, unsigned int time_limit, po::variables_map& vm) {
    int num_exp, num_gen;
    long long num_vec_comp;

    std::shared_ptr<AbstractSolver> solver;
    EPS eps_vec(graph.get_num_of_objectives(), eps);

    if (algorithm == "BOA"){
        solver = std::make_unique<BOAStarBucket>(graph, inv_graph, source, target);
    }else if (algorithm == "ABOA"){
        solver = std::make_unique<AnytimeBOA>(graph, inv_graph, source, target, 4);
    }else if (algorithm == "Apex"){
        solver = get_Astarpex_solver(graph, inv_graph, source, target, eps_vec);
    }else if (algorithm == "AnytimeApex"){
        if (vm["param1"].as<double>() > 0){
            decrease_factor = vm["param1"].as<double>();
        }
        solver = get_anytime_Apex_solver(graph, inv_graph, source, target, eps_vec, 2);
    }else if (algorithm == "AnytimeApexRestart"){
        if (vm["param1"].as<double>() > 0){
            decrease_factor = vm["param1"].as<double>();
        }
        solver = get_anytime_Apex_solver(graph, inv_graph, source, target, eps_vec, 3);
    }else if (algorithm == "AnytimeApexEnh"){
        if (vm["param1"].as<double>() > 0){
            decrease_factor =  vm["param1"].as<double>();
        }
        solver = get_anytime_Apex_solver(graph, inv_graph, source, target, eps_vec, 4);
    }else if (algorithm == "AnytimeApexHybrid"){
        if (vm["param1"].as<double>() > 0){
            decrease_factor = vm["param1"].as<double>();
        }
        if (vm["param2"].as<double>() > 0){
            default_hybrid_param =  vm["param2"].as<double>();
        }
        solver = get_anytime_Apex_solver(graph, inv_graph, source, target, eps_vec, 5);
    }else if (algorithm == "WCApex"){
        int bound = vm["bound"].as<int>();
        if (bound < 0){
            std::cerr << "no valid resource bound is given" << endl;
            exit(-1);
        }
        solver = std::make_unique <WCApexSearch>(graph, inv_graph, source, target, bound, eps);
    }else if (algorithm == "LTMOA"){
        solver = get_LTMOA_solver(graph, inv_graph, source, target);
    }else if (algorithm == "LTMOAeps"){
        solver = get_LTMOA_solver(graph, inv_graph, source, target, 0, eps_vec);
    }else if (algorithm == "LTMOAR"){
        use_R2 = true;
        solver = get_LTMOA2_solver(graph, inv_graph, source, target, false);
    }else if (algorithm == "LTMOAR1"){
        solver = get_LTMOA2_solver(graph, inv_graph, source, target, false);
    }else if (algorithm == "LTMOAR2"){
        use_R2 = true;
        solver = get_LTMOA_solver(graph, inv_graph, source, target);
    }else if (algorithm == "LTMOARBucket"){
        use_R2 = true;
        int param1 = 20000;
        if (vm["param1"].as<double>() > 0){
          param1 = (int) vm["param1"].as<double>();
        }
        bucket_step = param1;
        solver = get_LTMOA2_solver(graph, inv_graph, source, target, true);
    }else if (algorithm == "LTMOABucket"){
      int param1 = 20000;
      if (vm["param1"].as<double>() > 0){
        param1 = (int) vm["param1"].as<double>();
      }
        bucket_step = param1;
        solver = get_LTMOA_solver(graph, inv_graph, source, target, 1);
    // }else if (algorithm == "LTMOANDTree"){
    //     int param1 = 5;
    //     int param2 = 20;
    //     if (vm["param1"].as<double>() > 0){
    //         param1 = (int) vm["param1"].as<double>();
    //     }
    //     if (vm["param2"].as<double>() > 0){
    //         param2 = (int) vm["param2"].as<double>();
    //     }
    //     maxBranches = param1;
    //     maxListSize = param2;
    //     solver = get_LTMOA_solver(graph, inv_graph, source, target, 2);
    }else if (algorithm == "LazyLTMOA"){
        solver = get_LTMOA_solver(graph, inv_graph, source, target, -1);
    }else {
        exit(-1);
    }

    solver->verbal = vm["verbal"].as<int>();

    solver->start_time_ = std::chrono::steady_clock::now();
    solver->solve(time_limit);
    auto end = std::chrono::steady_clock::now();
    auto start = solver->start_time_;
    printf("Work took %f seconds\n", (end - start)/1.0s  );
    double runtime = ( end - start )/1.0s;

    std::cout << "Node expansion: " << solver->get_num_expansion() << std::endl;
    std::cout << "Runtime: " << runtime << std::endl;
    num_exp = solver->get_num_expansion();
    num_gen = solver->get_num_generation();
    num_vec_comp = __is_dominating_called_cnt__;

    if (log_sol){
        std::ostringstream stringStream;
        stringStream << dir_loc_sol << "/" << source << "_" << target << ".txt";
        log_sol_cost(stringStream.str(), solver->get_solution_log());
    }

    SolutionSet solutions = solver->get_solution_log();
    if (solutions.empty()) {
        output << algorithm << "," << source << "," << target << ",,,," << "\"\"" << "," << runtime << "," << num_gen << "," << num_exp << std::endl;
    } else {
        for (const auto &sol : solutions) {
            output << algorithm << "," << source << "," << target;
            for (int j = 0; j < 4; j++) {
                if (j < sol.cost.size()) {
                    output << "," << sol.cost[j];
                } else {
                    output << ",";
                }
            }
            output << ",\"";
            for (size_t j = 0; j < sol.path.size(); j++) {
                if (j > 0) output << ",";
                output << sol.path[j];
            }
            output << "\"," << runtime << "," << num_gen << "," << num_exp << std::endl;
        }
    }

    std::cout << "-----End Single Example-----" << std::endl;
}

void run_query(AdjacencyMatrix & graph, AdjacencyMatrix & inv_graph, std::string query_file, std::string output_file, std::string algorithm, double eps, int time_limit, po::variables_map& vm) {
    bool file_exists = boost::filesystem::exists(output_path + output_file);
    std::ofstream stats;
    stats.open(output_path + output_file, std::fstream::app);
    if (!file_exists) {
        stats << "solver,src,dest,distance,travel_time,elevation,avg_degree,path,runtime,num_generation,num_expansion\n";
    }

    std::vector<std::pair<size_t, size_t>> queries;
    if (load_query_pairs(query_file, queries) == false) {
        std::cout << "Failed to load queries file" << std::endl;
        return;
    }

    for (int i = vm["from"].as<int>(); i < std::min((int)queries.size(), vm["to"].as<int>()); i++){

        std::cout << "Started Query: " << i << "/" << std::min((int)queries.size(), vm["to"].as<int>()) << std::endl;
        size_t source = queries[i].first;
        size_t target = queries[i].second;
        if (source >= graph.size() || target >= graph.size()) {
            std::cerr << "Skipping invalid query: src=" << source << " tgt=" << target << " out of graph size " << graph.size() << std::endl;
            continue;
        }

        single_run_map(graph, inv_graph, source, target, stats, algorithm, eps, time_limit, vm);
    }
}

int main(int argc, char** argv){

    std::vector<string> objective_files;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("query,q", po::value<std::string>()->default_value(""), "the query file")
        ("query-csv", po::value<std::string>()->default_value(""), "the query CSV file with source,destination")
        ("from", po::value<int>()->default_value(0), "start from the i-th line of the query file")
        ("to", po::value<int>()->default_value(INT_MAX), "up to the i-th line of the query file")
        ("map,m",po::value< std::vector<string> >(&objective_files)->multitoken(), "files for edge weight")
        ("dataset-root", po::value<std::string>()->default_value(""), "DIMACS dataset root folder")
        ("region", po::value<std::string>()->default_value(""), "DIMACS region name (BAY,COL,FLA,NY)")
        ("output-dir", po::value<std::string>()->default_value(""), "output directory for batch DIMACS runs")
        ("algorithm,a", po::value<std::string>()->default_value("Apex"), "solvers (BOA, PPA or Apex search)")
        ("cutoffTime,t", po::value<int>()->default_value(300), "cutoff time (seconds)")
        ("bound", po::value<int>()->default_value(-1), "resource bound")
        ("logsolutions", po::value<std::string>()->default_value(""), "if non-empty, dump solution cost to the directory")
        ("output,o", po::value<std::string>(), "Name of the output file")
        ("eps,e", po::value<double>()->default_value(0), "epsilon values for approximate search")
        ("merge", po::value<std::string>()->default_value(""), "strategy for merging apex node pair: SMALLER_G2, RANDOM or MORE_SLACK")
        ("verbal", po::value<int>()->default_value(0), "level of the output details")
        ("param1", po::value<double>()->default_value(-1), "param 1")
        ("param2", po::value<double>()->default_value(-1), "param 2");


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    po::notify(vm);
    srand((int)time(0));

    std::string datasetRoot = vm["dataset-root"].as<std::string>();
    std::string queryCsv = vm["query-csv"].as<std::string>();
    std::string regionArg = vm["region"].as<std::string>();
    std::string outputDir = vm["output-dir"].as<std::string>();

    if (datasetRoot != "") {
        if (outputDir.empty()) {
            outputDir = "dataset";
            if (!boost::filesystem::exists(outputDir) && boost::filesystem::exists("../dataset")) {
                outputDir = "../dataset";
            }
        }
        boost::filesystem::create_directories(outputDir);

        std::vector<std::string> regions;
        if (regionArg != "") {
            regions.push_back(regionArg);
        } else {
            regions = {"NY", "COL", "BAY", "FLA"};
        }

        int timeLimit = vm["cutoffTime"].as<int>();
        double eps = vm["eps"].as<double>();
        std::vector<std::string> algorithms = {"Apex", "AnytimeApexHybrid"};

        for (const auto& region : regions) {
            std::string edgeFile = datasetRoot + "/edges_" + region + ".txt";
            size_t graphSize;
            std::vector<Edge> edges;
            if (!load_edge_txt_file(edgeFile, edges, graphSize)) {
                std::cerr << "Failed to load edge file for region " << region << ": " << edgeFile << std::endl;
                continue;
            }
            std::cout << "Loaded region " << region << " from " << edgeFile << ", graph size: " << graphSize << std::endl;

            AdjacencyMatrix graph(graphSize, edges);
            AdjacencyMatrix invGraph(graphSize, edges, true);

            std::vector<std::pair<size_t, size_t>> queries;
            if (vm["query"].as<std::string>() != "") {
                if (!load_query_pairs(vm["query"].as<std::string>(), queries)) {
                    std::cerr << "Failed to load query file: " << vm["query"].as<std::string>() << std::endl;
                    continue;
                }
            } else if (queryCsv != "") {
                if (!load_query_pairs(queryCsv, queries)) {
                    std::cerr << "Failed to load query CSV file: " << queryCsv << std::endl;
                    continue;
                }
            } else {
                std::string defaultQueryFile = datasetRoot + "/point_pairs_numeric.csv";
                if (!load_query_pairs(defaultQueryFile, queries)) {
                    std::cerr << "Failed to load default query file: " << defaultQueryFile << std::endl;
                    continue;
                }
            }

            if (queries.empty()) {
                std::cerr << "No queries loaded for region " << region << ". Skipping." << std::endl;
                continue;
            }

            for (const auto& alg : algorithms) {
                std::string outFile = outputDir + "/" + region + "_" + alg + ".txt";
                std::ofstream stats(outFile);
                if (!stats.is_open()) {
                    std::cerr << "Cannot create output file: " << outFile << std::endl;
                    continue;
                }
                stats << "solver,src,dest,distance,travel_time,elevation,avg_degree,path,runtime,num_generation,num_expansion\n";

                for (int i = vm["from"].as<int>(); i < std::min((int)queries.size(), vm["to"].as<int>()); i++) {
                    size_t source = queries[i].first;
                    size_t target = queries[i].second;
                    if (source >= graphSize || target >= graphSize) {
                        std::cerr << "Skipping invalid query for region " << region << ": src=" << source << " tgt=" << target << " out of graph size " << graphSize << std::endl;
                        continue;
                    }
                    std::cout << "Region " << region << ", algorithm " << alg << ", query " << i
                              << " (src=" << source << ", tgt=" << target << ")" << std::endl;
                    single_run_map(graph, invGraph, source, target, stats, alg, eps, timeLimit, vm);
                }
                stats.close();
            }
        }

        std::cout << "Batch mode completed." << std::endl;
        return 0;
    }

    if (vm["query"].as<std::string>() != ""){
        if (vm["start"].as<int>() != -1 || vm["goal"].as<int>() != -1){
            std::cerr << "query file and start/goal cannot be given at the same time !" << std::endl;
            return -1;
        }
    }
    
    dir_loc_sol = vm["logsolutions"].as<std::string>();
    log_sol = dir_loc_sol.size()> 0;
    if (log_sol){
        boost::filesystem::create_directory(dir_loc_sol);
    }

    size_t graph_size;
    std::vector<Edge> edges;

    for (auto file:objective_files){
        std::cout << file << std::endl;
    }

    if (load_gr_files(objective_files, edges, graph_size) == false) {
        std::cout << "Failed to load gr files" << std::endl;
        return -1;
    }

    std::cout << "Graph Size: " << graph_size << std::endl;

    // Build graphs
    AdjacencyMatrix graph(graph_size, edges);
    AdjacencyMatrix inv_graph(graph_size, edges, true);


    if (vm["query"].as<std::string>() != ""){
        run_query(graph, inv_graph, vm["query"].as<std::string>(), vm["output"].as<std::string>(), vm["algorithm"].as<std::string>(), vm["eps"].as<double>(), vm["cutoffTime"].as<int>(), vm);
    } else{
        std::ofstream stats;
        stats.open(vm["output"].as<std::string>(), std::fstream::app);

        single_run_map(graph, inv_graph, vm["start"].as<int>(), vm["goal"].as<int>(), stats, vm["algorithm"].as<std::string>(), vm["eps"].as<double>(), vm["cutoffTime"].as<int>(), vm);
    }

    return 0;
}
