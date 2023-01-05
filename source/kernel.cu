/* Program to simulate the Bornholdt Ising Model (https://arxiv.org/pdf/cond-mat/0105224.pdf)
 *
 * The program reads in a configuration file "multising.conf" or first command line parameter. The
 * configuration file contains the parameter choice for the simulation.
 *
 * Example configuration file:
 *
 * lattice_height = 8192           # Lattice size
 * lattice_width = 8192            # Lattice size
 * total_updates = 10000           # Number of iterations to perform
 * seed = 1591361                  # Seed for the simulation
 * alpha = 128.00                  # Parameter of the model (coupling strength to the magnetisation)
 * j = 1.0                         # Parameter of the model (coupling strength to the neighbours)
 * beta = 1.0                      # Parameter of the model (pseudo-temperature)
 * init_up = 0.5                   # Percentage of spins initially pointing up
 * rng_offset = 124837             # Used to resume the simulation at given time point in combination with import (optional)
 * import = iteration_124837.dat   # Used to resume simulation with given state in file (optional)
 * export = final_state.dat        # Save the final configuration to file with specified name (optional)
 *
 * By default, the relative magnetisation will be saved in a file magnetisation_*.dat where a new file every
 * FILE_ENTRY_LIMIT (1e6) iterations will be created.
 * To save the lattice configuration edit the if clause in the main update loop.
 *
 * At the end of the simulation an additional line will be added to the configuration file denoting the
 * reached number of iterations.
 *
 * final_iteration = 10000
 *
 * This can be used together with the exported final_state to resume the simulation.
 *
 */
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <csignal>

#include "cudamacro.h"
#include "traders.cuh"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCDFAInspection"
using std::string;

#define DIV_UP(a,b)  (((a) + ((b) - 1)) / (b))
#define MIN(a,b)	(((a) < (b)) ? (a) : (b))

#define THREADS (128)
#define BIT_X_SPIN (4)

#define THREADS_X (16)
#define THREADS_Y (16)

#define FILE_ENTRY_LIMIT (1000000)


// trim from left
inline string& ltrim(string& s, const char* t = " \t\n\r\f\v") {
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from right
inline string& rtrim(string& s, const char* t = " \t\n\r\f\v") {
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from left & right
inline string& trim(string& s, const char* t = " \t\n\r\f\v") {
    return ltrim(rtrim(s, t), t);
}

volatile sig_atomic_t flag_terminate = 0;
void sigint(int sig) {  // can be called asynchronously
    flag_terminate = 1; // set flag
}


/**
 * @brief Read in the simulation details from a configuration file
 *
 * Numerous parameters need to be passed to the simulation via a configuration file in which fields and values
 * are separated by an equal sign.
 *
 * @param config_filename  The name of the configuration file
 * @param delimiter  The delimiter, or "assigment operator", for fields and values. Defaults to "="
 * @return  A map containing the fields and values read from the configuration file
 *
 */
std::map<string, string> readConfigFile(const char* config_filename, const string& delimiter = "=") {
    std::ifstream config_file;
    config_file.open(config_filename);
    std::map<string, string> config;

    if (!config_file.is_open()) {
        std::cout << "Could not open config file '" << config_filename << "'" << std::endl;
        exit(EXIT_FAILURE);
    }

    int row = 0;
    string line;
    string key;

    while (getline(config_file, line)) {
        if (line[0] == '#' || line.empty()) continue;
        int delimiter_position = line.find(delimiter);

        for (int idx = 0; idx < delimiter_position; idx++) {
            if (line[idx] != ' ') key += line[idx];
        }

        string value = line.substr(delimiter_position + 1, line.length() - 1);
        config[key] = value;
        row++;
        key = "";
    }
    config_file.close();
    return config;
}


void validateGrid(const long long lattice_width, const long long lattice_height,
                  const int spin_x_word) {
    if (!lattice_width || (lattice_width % 2) || ((lattice_width / 2) % (2 * spin_x_word * THREADS_X))) {
        fprintf(stderr, "\nPlease specify an lattice_width multiple of %d\n\n",
                2 * spin_x_word * 2 * THREADS_X);
        exit(EXIT_FAILURE);
    }
    if (!lattice_height || (lattice_height % (THREADS_Y))) {
        fprintf(stderr, "\nPlease specify a lattice_height multiple of %d\n\n", THREADS_Y);
        exit(EXIT_FAILURE);
    }
}


cudaDeviceProp identifyGpu() {
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
    unsigned long long *d_sum;

    string import_file;
    string export_file;
    bool read_from_file = false;
    bool dump_to_file = false;

    cudaEvent_t start, stop;
    float elapsed_time;

    std::ofstream mag_file;
    Parameters params;

    const char *config_filename = (argc == 1) ? "multising.conf" : argv[1];
    std::map<string, string> config = readConfigFile(config_filename);

    params.lattice_height = std::stoll(config["lattice_height"]);
    params.lattice_width = std::stoll(config["lattice_width"]);
    params.seed = std::stoull(config["seed"]);
    const unsigned int total_updates = std::stoul(config["total_updates"]);
    float alpha = std::stof(config["alpha"]);
    float j = std::stof(config["j"]);
    float beta = std::stof(config["beta"]);
    float percentage_up = std::stof(config["init_up"]);

    if (config.count("rng_offset")) {
        params.rng_offset = std::stoull(config["rng_offset"]);
    } else {
        params.rng_offset = 0;
    }
    if (config.count("import")) {
        import_file = config["import"];
        trim(import_file);
        read_from_file = true;
        std::cout << "Using existing lattice state from file: " << import_file << std::endl;
    }
    if (config.count("export")) {
        export_file = config["export"];
        trim(export_file);
        dump_to_file = true;
    }

    params.reduced_alpha = -2.0f * beta * alpha;
    params.reduced_j = -2.0f * beta * j;

    validateGrid(params.lattice_width, params.lattice_height, SPIN_X_WORD);
    cudaDeviceProp props = identifyGpu();

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

    if (read_from_file) {
        std::cout << "Reading in lattice configuration..." << std::endl;
        readFromFileBinary(d_spins, import_file.c_str(), params.lattice_height,
                     params.words_per_row, params.total_words);
    } else {
        // words_per_row / 2 because words two 64 bit words are compacted into
        // one 128 bit word
        std::cout << "Initialising random lattice state..." << std::endl;
        initialiseArrays<unsigned long long>(
                blocks, threads_per_block,
                params.seed, params.words_per_row / 2,
                d_black_tiles, d_white_tiles, percentage_up
        );
    }

    CHECK_CUDA(cudaSetDevice(0))
    CHECK_CUDA(cudaDeviceSynchronize())

    mag_file.open("magnetisation_" + std::to_string(params.rng_offset) + ".dat");
    int iteration;
    float relative_magnetisation;
    signal(SIGINT, sigint);
    CHECK_CUDA(cudaEventRecord(start, nullptr))
    for(iteration = params.rng_offset; iteration < total_updates; iteration++) {
        relative_magnetisation = update(
                iteration, blocks, threads_per_block, reduce_blocks,
                d_black_tiles, d_white_tiles, d_sum, d_probabilities,
                params
        );
        mag_file << relative_magnetisation << std::endl;

        // create a new file every FILE_ENTRY_LIMIT iterations
        if (iteration % FILE_ENTRY_LIMIT == 0 && iteration) {
            mag_file.close();
            mag_file.open("magnetisation_" + std::to_string(iteration) + ".dat");
        }

        if (iteration % 30000000 == 0 && iteration) {
            FILE *f = fopen("backup.info", "w");
            fprintf(f, "Backup of iteration %d\n", iteration);
            fclose(f);
            dumpLatticeBinary("lattice_backup.bin", params.lattice_height, params.words_per_row,
                              params.total_words, d_spins);
        }
        if (flag_terminate) {
            std::cout << "Received keyboard interrupt, exiting..." << std::endl;
            break;
        }
    }
    mag_file.close();
    CHECK_CUDA(cudaEventRecord(stop, nullptr))
    CHECK_CUDA(cudaEventSynchronize(stop))

    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop))
    double spin_updates_per_nanosecond = static_cast<double>(params.total_words * SPIN_X_WORD) * iteration / (elapsed_time * 1.0E+6);
    std::cout << "Computation time: " << elapsed_time * 1.0E-3 << "s" << std::endl;
    std::cout << "Updates per ns: " << spin_updates_per_nanosecond << std::endl;
    if (dump_to_file) {
        std::cout << "Saving lattice state for reuse..." << std::endl;
        dumpLatticeBinary(export_file.c_str(), params.lattice_height, params.words_per_row,
                    params.total_words, d_spins);
    }
    FILE *fp = fopen(config_filename, "a");
    fprintf(fp, "final_iteration = %d\n", iteration);
    fclose(fp);
    CHECK_CUDA(cudaFree(d_spins))
    CHECK_CUDA(cudaFree(d_probabilities))
    CHECK_CUDA(cudaFree(d_sum))
    CHECK_CUDA(cudaSetDevice(0))
    CHECK_CUDA(cudaDeviceReset())
    return 0;
}
