#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <map>

#include "cudamacro.h"
#include "traders.cuh"

using namespace std;

#define DIV_UP(a,b)  (((a) + ((b) - 1)) / (b))
#define MIN(a,b)	(((a) < (b)) ? (a) : (b))

#define THREADS (128)
#define BIT_X_SPIN (4)

#define THREADS_X (16)
#define THREADS_Y (16)

#define FILE_ENTRY_LIMIT (1000000)


map<string, string> read_config_file(const string& config_filename, const string& delimiter = "=") {
    std::ifstream config_file;
    config_file.open(config_filename);
    map<string, string> config;

    if (!config_file.is_open()) {
        std::cout << "Could not open config file '" << config_filename << "'" << std::endl;
        exit(EXIT_FAILURE);
    }

    int row = 0;
    std::string line;
    std::string key;

    while (getline(config_file, line)) {
        if (line[0] == '#' || line.empty()) continue;
        int delimiter_position = line.find(delimiter);

        for (int idx = 0; idx < delimiter_position; idx++) {
            if (line[idx] != ' ') key += line[idx];
        }

        std::string value = line.substr(delimiter_position + 1, line.length() - 1);
        config[key] = value;
        row++;
        key = "";
    }
    config_file.close();
    return config;
}


void validate_grid(const long long lattice_width, const long long lattice_height,
                   const int spin_x_word) {
    if (!lattice_width || (lattice_width % 2) || ((lattice_width / 2) % (2 * spin_x_word * THREADS_X))) {
        fprintf(stderr, "\nPlease specify an lattice_width multiple of %d\n\n", 2 * spin_x_word * 2 * THREADS_X);
        exit(EXIT_FAILURE);
    }
    if (!lattice_height || (lattice_height % (THREADS_Y))) {
        fprintf(stderr, "\nPlease specify a lattice_height multiple of %d\n\n", THREADS_Y);
        exit(EXIT_FAILURE);
    }
}


cudaDeviceProp identify_gpu() {
    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0))
    /*
    printf("\nUsing GPU: %s, %d SMs, %d th/SM max, CC %d.%d, ECC %s\n",
    props.name, props.multiProcessorCount,
    props.maxThreadsPerMultiProcessor,
    props.major, props.minor,
    props.ECCEnabled ? "on" : "off");
    */
    return props;
}


int main(int argc, char **argv) {
    unsigned long long *d_spins = nullptr;
    const int SPIN_X_WORD = (8 * sizeof(*d_spins)) / BIT_X_SPIN;
    unsigned long long *d_black_tiles;
    unsigned long long *d_white_tiles;

    unsigned long long spins_up;
    unsigned long long spins_down;
    unsigned long long *d_sum;

    cudaEvent_t start, stop;
    float elapsed_time;

    std::ofstream mag_file;
    Parameters params;

    string config_filename = (argc == 1) ? "multising.conf" : argv[1];
    map<string, string> config = read_config_file(config_filename);

    params.lattice_height = std::stoll(config["lattice_height"]);
    params.lattice_width = std::stoll(config["lattice_width"]);
    params.seed = std::stoull(config["seed"]);
    const unsigned int total_updates = std::stoul(config["total_updates"]);
    float alpha = std::stof(config["alpha"]);
    float j = std::stof(config["j"]);
    float beta = std::stof(config["beta"]);
    float percentage_up = std::stof(config["init_up"]);

    params.reduced_alpha = -2.0f * beta * alpha;
    params.reduced_j = -2.0f * beta * j;

    validate_grid(params.lattice_width, params.lattice_height, SPIN_X_WORD);
    cudaDeviceProp props = identify_gpu();

    params.words_per_row = (params.lattice_width / 2) / SPIN_X_WORD;
    params.total_words = 2ull * static_cast<size_t>(params.lattice_height) * params.words_per_row;

    // words_per_row / 2 because each entry in the array has two components
    dim3 blocks(DIV_UP(params.words_per_row / 2, THREADS_X),
                DIV_UP(params.lattice_height, THREADS_Y));
    dim3 threads_per_block(THREADS_X, THREADS_Y);
    const int reduce_blocks = MIN(DIV_UP(params.total_words, THREADS),
                                  (props.maxThreadsPerMultiProcessor / THREADS) * props.multiProcessorCount);

    CHECK_CUDA(cudaMalloc(&d_spins, params.total_words * sizeof(*d_spins)))
    CHECK_CUDA(cudaMemset(d_spins, 0, params.total_words * sizeof(*d_spins)))

    CHECK_CUDA(cudaMalloc(&d_sum, 2 * sizeof(*d_sum)))

    d_black_tiles = d_spins;
    d_white_tiles = d_spins + params.total_words / 2;

    float *d_probabilities;
    CHECK_CUDA(cudaMallocPitch(&d_probabilities, &params.pitch,
                               5 * sizeof(*d_probabilities), 2))

    CHECK_CUDA(cudaEventCreate(&start))
    CHECK_CUDA(cudaEventCreate(&stop))

    // words_per_row / 2 because words two 64 bit words are compacted into
    // one 128 bit word
    initialise_arrays<unsigned long long>(
            blocks, threads_per_block,
            params.seed, params.words_per_row / 2,
            d_black_tiles, d_white_tiles, percentage_up
    );

    CHECK_CUDA(cudaSetDevice(0))
    CHECK_CUDA(cudaDeviceSynchronize())

    mag_file.open("magnetisation_0.dat");
    int iteration;
    float global_market;
    CHECK_CUDA(cudaEventRecord(start, nullptr))
    for(iteration = 0; iteration < total_updates; iteration++) {
        global_market = update(
            iteration, blocks, threads_per_block, reduce_blocks,
            d_black_tiles, d_white_tiles, d_sum, d_probabilities,
            spins_up, spins_down, params
        );
        mag_file << global_market << std::endl;

        // create a new file every FILE_ENTRY_LIMIT iterations
        if (iteration % FILE_ENTRY_LIMIT == 0 && (iteration)) {
            mag_file.close();
            mag_file.open("magnetisation_" + std::to_string(iteration) + ".dat");
        }

        //if (iteration % 50 == 0)
        //   dumpLattice(iteration, params.lattice_height, params.words_per_row,
        //              params.total_words, d_spins);
    }
    mag_file.close();
    CHECK_CUDA(cudaEventRecord(stop, nullptr))
    CHECK_CUDA(cudaEventSynchronize(stop))

    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop))
    double spin_updates_per_nanosecond = static_cast<double>(params.total_words * SPIN_X_WORD) * iteration / (elapsed_time * 1.0E+6);
    std::cout << "Beta: " << beta << std::endl;
    std::cout << "Computation time: " << elapsed_time * 1.0E-3 << "s" << std::endl;
    std::cout << "Updates per ns: " << spin_updates_per_nanosecond << std::endl;
    CHECK_CUDA(cudaFree(d_spins))
    CHECK_CUDA(cudaFree(d_probabilities))
    CHECK_CUDA(cudaFree(d_sum))
    CHECK_CUDA(cudaSetDevice(0))
    CHECK_CUDA(cudaDeviceReset())
    return 0;
}