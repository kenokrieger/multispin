#ifndef traders_cuh
#define traders_cuh

#include <curand_kernel.h>
#include "cudamacro.h"

#define BIT_X_SPIN (4)
#define THREADS (128)

#define BLOCK_DIMENSION_X_DEFINE (16)
#define BLOCK_DIMENSION_Y_DEFINE (16)

enum {C_BLACK, C_WHITE};

/**
 * @brief Contains the different parameters for the simulation
 *
 * @field seed The seed for the random number generator
 * @field reduced_alpha - The parameter alpha multiplied by -2 times beta
 * @field lattice_height - The desired height of the lattice
 * @field lattice_width - The desired width of the lattice
 * @field words_per_row - The number of computer words per row as a result of the chosen configuration
 * @field total_words - The total number of words
 * @field pitch - The pitch of the precomputed probabilities which is needed in the call to cudaMemcpy2D()
 * @field rng_offset - An offset that can be passed to the random number generator to resume a simulation
 *
 */
typedef struct {
    unsigned long long seed;
    float reduced_alpha;
    float reduced_j;
    long long lattice_height;
    long long lattice_width;
    size_t words_per_row;
    size_t total_words;
    size_t pitch;
    size_t rng_offset;
} Parameters;

// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
__device__ __forceinline__ unsigned long long int custom_popc(const unsigned long long int x) {return __popcll(x);}

// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
__device__ __forceinline__ ulonglong2 custom_make_int2(const unsigned long long x, const unsigned long long y) {
    return make_ulonglong2(x, y);
}

/**
 * @brief Initialise the entries in an array to random bits.
 *
 * Given the pointer to an array, randomly set every fourth bit of each array element to 0 or 1.
 *
 * @tparam BITXSPIN The number of bits allocated for each spin
 * @tparam COLOR The color of the tiles contained in the array according to the checkerboard algorithm
 * @tparam INT_T The type of the array elements
 *
 * @param seed The seed for the random number generator
 * @param ncolumns Number of columns of the array
 * @param agents The pointer to the array to initialise
 * @param percentage The probability for assigning 1 to an element. Defaults to 0.5
 *
 */
template<int BITXSPIN, int COLOR, typename INT_T, typename INT2_T>
__global__  void initialiseAgents(const unsigned long long seed,
                                  const long long ncolumns,
                                  INT2_T *__restrict__ agents,
                                  float percentage = 0.5f) {
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto index = row * ncolumns + col;
    const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSPIN;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, index, static_cast<long long>(2 * SPIN_X_WORD) * COLOR, &rng);

    agents[index] = custom_make_int2(INT_T(0), INT_T(0));
    for(int spin_position = 0; spin_position < 8 * sizeof(INT_T); spin_position += BITXSPIN) {
        // Initialise x and y component with random spins
        if (curand_uniform(&rng) < percentage) {
            agents[index].x |= INT_T(1) << spin_position;
        }
        if (curand_uniform(&rng) < percentage) {
            agents[index].y |= INT_T(1) << spin_position;
        }
    }
}

/**
 * @brief Initialise the entries in two arrays by calling initialiseAgents
 *
 * Given the pointer to two arrays, randomly set every fourth bit of each array element to 0 or 1.
 *
 * @tparam INT_T The type of the array elements
 *
 * @param blocks The number of blocks for the launch on the GPU
 * @param threads_per_block The number of threads for the launch on the GPU
 * @param seed The seed for the random number generator
 * @param ncolumns The number of columns in each array
 * @param d_black_tiles The pointer to the "black tiles" of the lattice according to the checkerboard algorithm
 * @param d_white_tiles The pointer to the "white tiles" of the lattice according to the checkerboard algorithm
 * @param percentage The percentage of spins to be initialised with 1
 *
 */
template<typename INT_T, typename INT2_T>
void initialiseArrays(dim3 blocks, dim3 threads_per_block,
                      const unsigned long long seed, const unsigned long long ncolumns,
                      INT2_T *__restrict__ d_black_tiles, INT2_T *__restrict__ d_white_tiles,
                      float percentage = 0.5f) {
    initialiseAgents<BIT_X_SPIN, C_BLACK, INT_T><<<blocks, threads_per_block>>>(
            seed, ncolumns, reinterpret_cast<ulonglong2 *>(d_black_tiles), percentage
    );
    CHECK_ERROR("initialiseAgents")

    initialiseAgents<BIT_X_SPIN, C_WHITE, INT_T><<<blocks, threads_per_block>>>(
            seed, ncolumns, reinterpret_cast<ulonglong2 *>(d_white_tiles), percentage
    );
    CHECK_ERROR("initialiseAgents")
}

/**
 * @brief Loads a selection from an array into the shared memory of the GPU
 *
 * Given a pointer to a device array, load a fraction of the array defined by BLOCK_SIZE_X and
 * BLOCK_SIZE_Y into the shared memory of the GPU. The array in the shared memory has size
 * BLOCK_SIZE_X + 2 x BLOCK_SIZE_Y + 2 where the extra two rows/columns are used to take care of boundary
 * conditions.
 *
 * Example:
 *
 * Source lattice [11 x 16]                             Imported tile [6 x 6]
 *                   |-------------| <- Tile Border       1 0 1 0 1 0   <- extra rows and columns to take
 * 1 0 0 0 1 0 1 1 0 | 0 1 1 0 1 0 |                    0 0 1 1 0 1 0 1    care of boundary conditions
 * 0 1 0 0 1 0 0 1 0 | 1 0 1 0 0 1 |                    0 1 0 1 0 0 1 0
 * 0 0 1 0 1 0 1 0 1 | 1 0 1 0 0 1 |                    1 1 0 1 0 0 1 0
 * 1 1 1 0 0 1 0 1 1 | 0 0 1 0 1 0 |                    1 0 0 1 0 1 0 1
 * 1 0 0 1 1 0 1 0 1 | 1 1 1 0 1 0 |                    1 1 1 1 0 1 0 1
 * 1 1 1 0 1 0 1 1 0 | 0 1 1 0 1 1 |                    0 0 1 1 0 1 1 1
 * 1 1 0 1 0 1 0 0 0 |-------------|                      1 0 1 1 1 1
 * 0 1 0 1 1 0 0 1 1 0 1 0 1 1 1 1
 * 0 1 0 1 0 1 0 0 0 1 1 0 1 0 0 0                      The extra row at the top contains the upper neighbours of the
 * 1 1 1 0 0 1 0 1 0 1 0 1 0 0 1 0                      spins in the 6 x 6 tile. In this case these are contained in
 * 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0                      the last row of the lattice (periodic boundary conditions)
 *
 *
 * This tile setup has the advantage that in the neighbour sum computation boundary conditions are automatically
 * satisfied by reading in neighbour spins.
 *
 * @tparam BLOCK_SIZE_X The x dimension of the block which is loaded into shared memory
 * @tparam BLOCK_SIZE_Y The y dimension of the block which is loaded into shared memory
 * @tparam INT2_T The type of the array elements
 *
 * @param ncolumns The number of columns in the array to load in
 * @param nrows The number of rows in the array to load in
 * @param agents The pointer to the array to be loaded in
 * @param tile The pointer to array located in the shared memory
 *
 */
template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y, typename INT2_T>
__device__ void loadTiles(const int ncolumns, const int nrows,
                          const INT2_T *__restrict__ agents, INT2_T tile[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2]) {
    const unsigned int tidx = threadIdx.x;
    const unsigned int tidy = threadIdx.y;

    const int tile_start_x = blockIdx.x * BLOCK_SIZE_X;
    const int tile_start_y = blockIdx.y * BLOCK_SIZE_Y;

    int row = tile_start_y + tidy;
    int col = tile_start_x + tidx;
    tile[1 + tidy][1 + tidx] = agents[row * ncolumns + col];

    if (tidy == 0) {
        row = (tile_start_y % nrows) == 0 ? nrows - 1 : tile_start_y - 1;
        tile[0][1 + tidx] = agents[row * ncolumns + col];

        row = (tile_start_y + BLOCK_SIZE_Y) % nrows;
        tile[1 + BLOCK_SIZE_Y][1 + tidx] = agents[row * ncolumns + col];

        row = tile_start_y + tidx;
        col = (tile_start_x % ncolumns) == 0 ? ncolumns - 1 : tile_start_x - 1;
        tile[1 + tidx][0] = agents[row * ncolumns + col];

        row = tile_start_y + tidx;
        col = (tile_start_x + BLOCK_SIZE_X) % ncolumns;
        tile[1 + tidx][1 + BLOCK_SIZE_X] = agents[row * ncolumns + col];
    }
}

/**
 * @brief Loads an array of precomputed spin orientation possibilities into shared memory.
 *
 * @param precomputed_probabilities The pointer to the array of precomputed probabilities
 * @param shared_probabilities The pointer to the array located in the shared memory of the GPU
 *
 */
__device__ void loadProbabilities(const float precomputed_probabilities[][5], float shared_probabilities[][5]) {
    const unsigned tidx = threadIdx.x;
    const unsigned tidy = threadIdx.y;
    // load precomputed probabilities into shared memory
    // loops are for cases in which the block dimension is smaller than the array size
    #pragma unroll
    for(unsigned i = 0; i < 2; i += blockDim.y) {
        #pragma unroll
        for(unsigned j = 0; j < 5; j += blockDim.x) {
            if (i + tidy < 2 && j + tidx < 5)
                shared_probabilities[i + tidy][j + tidx] = precomputed_probabilities[i + tidy][j + tidx];
        }
    }
}

/**
 * @brief Compute the neighbour sum for a spin at a given position
 *
 * Given a spin position, locate the nearest neighbours of that spin as they are given in the Multispin coding
 * approach and sum over them.
 *
 * @tparam BLOCK_DIMENSION_X The x dimension of the block
 * @tparam BITXSPIN The number of bits allocated for each spin
 * @tparam INT2_T The type of the array elements
 *
 * @param shared_tiles Pointer to the array containing the neighbours
 * @param tidx X position of the spin(s)
 * @param tidy Y position of the spin(s)
 * @param shift_left Boolean specifying whether the nearest neighbours are obtained by bit shifting one neighbour to
 *                     the left or right
 *
 * @returns The calculated neighbour sum stored bitwise in a computer word.
 *
 */
template<int BLOCK_DIMENSION_X, int BITXSPIN, typename INT_T, typename INT2_T>
__device__ INT2_T computeNeighbourSum(INT2_T shared_tiles[][BLOCK_DIMENSION_X + 2],
                                      const int tidx, const int tidy, const int shift_left) {
    // read in nearest neighbours from the loaded in sub lattice (tile)
    INT2_T upper_neighbor  = shared_tiles[    tidy][1 + tidx];
    INT2_T center_neighbor = shared_tiles[1 + tidy][1 + tidx];
    INT2_T lower_neighbor  = shared_tiles[2 + tidy][1 + tidx];

    // remaining neighbor, either left or right
    INT2_T horizontal_neighbor = (shift_left) ? shared_tiles[1 + tidy][tidx] : shared_tiles[1 + tidy][2 + tidx];

    if (shift_left) {
        horizontal_neighbor.x = (center_neighbor.x << BITXSPIN) | (horizontal_neighbor.y >> (8 * sizeof(horizontal_neighbor.y) - BITXSPIN));
        horizontal_neighbor.y = (center_neighbor.y << BITXSPIN) | (center_neighbor.x >> (8 * sizeof(center_neighbor.x) - BITXSPIN));
    } else {
        horizontal_neighbor.y = (center_neighbor.y >> BITXSPIN) | (horizontal_neighbor.x << (8 * sizeof(horizontal_neighbor.x) - BITXSPIN));
        horizontal_neighbor.x = (center_neighbor.x >> BITXSPIN) | (center_neighbor.y << (8 * sizeof(center_neighbor.y) - BITXSPIN));
    }

    // this basically sums over all spins/word in parallel
    center_neighbor.x += upper_neighbor.x + lower_neighbor.x + horizontal_neighbor.x;
    center_neighbor.y += upper_neighbor.y + lower_neighbor.y + horizontal_neighbor.y;

    return center_neighbor;
}

/**
 * @brief Flip the spins stored in a computer word according to the system dynamics
 *
 * Update the spin orientation of a spins stored in a computer word according to its orientation
 * probability
 *
 * @tparam BITXSPIN  The number of bits allocated for each spin
 * @tparam INT2_T  The type of the array entries
 *
 * @param rng  The random number generator
 * @param target  The spin(s) to update
 * @param parallel_sum  The calculated neighbour sum
 * @param  shared_probabilities  The pointer to the precomputed spin orientation probabilities stored in the shared
 *                               memory
 *
 * @returns The array entry where the orientation of the individual spins has been updated
 */
template<int BITXSPIN, typename INT_T, typename INT2_T>
__device__ INT2_T flipSpins(curandStatePhilox4_32_10_t rng, INT2_T target, INT2_T parallel_sum,
                            const float shared_probabilities[][5]) {
    const auto ONE = static_cast<INT_T>(1);
    #pragma unroll
    for(int spin_position = 0; spin_position < 8 * sizeof(INT_T); spin_position += BITXSPIN) {
        const int2 spin = make_int2((target.x >> spin_position) & 0xF, (target.y >> spin_position) & 0xF);
        const int2 sum = make_int2((parallel_sum.x >> spin_position) & 0xF,
                                   (parallel_sum.y >> spin_position) & 0xF);

        if (curand_uniform(&rng) <= shared_probabilities[spin.x][sum.x])
            target.x |= (ONE << spin_position);
        else
            target.x &= ~(ONE << spin_position);

        if (curand_uniform(&rng) <= shared_probabilities[spin.y][sum.y])
            target.y |= (ONE << spin_position);
        else
            target.y &= ~(ONE << spin_position);
    }
    return target;
}

/**
 * @brief Update all spins in a given array according to the system dynamics
 *
 * Update the spin orientation of a spins stored bitwise in an array by calculating their neighbour sum
 * and changing their value according to precomputed probabilities
 *
 * @tparam BLOCK_DIMENSION_X  The x dimension of the block on the GPU
 * @tparam BLOCK_DIMENSION_Y  The y dimension of the block on the GPU
 * @tparam BITXSPIN  The number of bits allocated for each spin
 * @tparam COLOR The color of the tiles contained in the array according to the checkerboard algorithm
 * @tparam INT2_T  The type of the array entries
 *
 * @param seed  The seed for the random number generator
 * @param rng_invocations  The number of previous calls to the random number generator
 * @param ncolumns  The number of columns in the array
 * @param nrows  The number of rows in the array
 * @param precomputed_probabilities  The pointer to the array of precomputed probabilities stored in the shared
 *                                   memory of the GPU
 * @param checkerboard_agents  The pointer to the array containing the neighbour spins
 * @param agents  The pointer to the array containing the spins to update
 *
 */
template<int BLOCK_DIMENSION_X, int BLOCK_DIMENSION_Y, int BITXSPIN, int COLOR, typename INT_T, typename INT2_T>
__global__ void updateStrategies(const unsigned long long seed, const int rng_invocations,
                                 const size_t ncolumns,
                                 const size_t nrows,
                                 const float precomputed_probabilities[][5],
                                 const INT2_T *__restrict__ checkerboard_agents,
                                 INT2_T *__restrict__ agents) {
    const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSPIN;
    const unsigned tidx = threadIdx.x;
    const unsigned tidy = threadIdx.y;
    __shared__ INT2_T shared_tiles[BLOCK_DIMENSION_Y + 2][BLOCK_DIMENSION_X + 2];
    loadTiles<BLOCK_DIMENSION_X, BLOCK_DIMENSION_Y, INT2_T>(ncolumns, nrows,
                                                            checkerboard_agents, shared_tiles);
    __shared__ float shared_probabilities[2][5];
    loadProbabilities(precomputed_probabilities, shared_probabilities);
    __syncthreads();

    int row = blockIdx.y * BLOCK_DIMENSION_Y + tidy;
    int col = blockIdx.x * BLOCK_DIMENSION_X + tidx;
    const int shift_left = (COLOR == C_BLACK) == !(row % 2);
    const size_t index = row * ncolumns + col;
    // compute neighbor sum
    INT2_T parallel_sum = computeNeighbourSum<BLOCK_DIMENSION_X, BITXSPIN, INT_T>(shared_tiles, tidx, tidy,
                                                                                  shift_left);
    // flip spin according to neighbor sum and its own orientation
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, index, static_cast<long long>(2 * SPIN_X_WORD) * (2 * rng_invocations + COLOR),
                &rng);
    agents[index] = flipSpins<BITXSPIN, INT_T>(rng, agents[index], parallel_sum, shared_probabilities);
}

/**
 * @brief Precompute the possible spin orientation probabilities
 *
 * @param probabilities  The pointer to the array to be filled with the precomputed probabilities
 * @param market_coupling  The second term in the local field multiplied by -2 times beta times alpha
 *                         market_coupling = -2 * beta * alpha * relative magnetisation
 * @param reduced_j  The parameter j multiplied by -2 times beta
 * @param pitch  The pitch needed for the call to cudaMemcpy2D
 *
 */
void precomputeProbabilities(float* probabilities, const float market_coupling, const float reduced_j, const size_t pitch) {
    float h_probabilities[2][5];

    for (int spin = 0; spin < 2; spin++) {
        for (int idx = 0; idx < 5; idx++) {
            int neighbour_sum = 2 * idx - 4;
            float field = reduced_j * neighbour_sum - market_coupling * (-1 + 2 * spin);
            h_probabilities[spin][idx] = 1.0 / (1.0 + exp(field));
        }
    }
    CHECK_CUDA(cudaMemcpy2D(probabilities, 5 * sizeof(*h_probabilities), &h_probabilities, pitch,
                            5 * sizeof(*h_probabilities), 2, cudaMemcpyHostToDevice))
}


// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
template<int BLOCK_DIMENSION_X, int WSIZE, typename T>
__device__ __forceinline__ T blockSum(T traders)
{
    __shared__ T sh[BLOCK_DIMENSION_X / WSIZE];

    const int lid = threadIdx.x % WSIZE;
    const int wid = threadIdx.x / WSIZE;

    #pragma unroll
    for(int i = WSIZE/2; i; i >>= 1) {
        traders += __shfl_down_sync(0xFFFFFFFF, traders, i);
    }
    if (lid == 0) sh[wid] = traders;

    __syncthreads();
    if (wid == 0) {
        traders = (lid < (BLOCK_DIMENSION_X / WSIZE)) ? sh[lid] : 0;

        #pragma unroll
        for(int i = (BLOCK_DIMENSION_X/WSIZE)/2; i; i >>= 1) {
            traders += __shfl_down_sync(0xFFFFFFFF, traders, i);
        }
    }
    __syncthreads();
    return traders;
}


// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
template<int BLOCK_DIMENSION_X, int BITXSPIN, typename INT_T, typename SUM_T>
__global__ void computeMagnetisation(const size_t n,
                                     const INT_T *__restrict__ traders,
                                     SUM_T *__restrict__ sum)
{
    // to be optimized
    const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSPIN;

    const long long nth = static_cast<long long>(blockDim.x) * gridDim.x;
    const long long thread_id = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    SUM_T cntP = 0;
    SUM_T cntN = 0;

    for(long long i = 0; i < n; i += nth) {
        if (i + thread_id < n) {
            const int c = custom_popc(traders[i + thread_id]);
            cntP += c;
            cntN += SPIN_X_WORD - c;
        }
    }
    cntP = blockSum<BLOCK_DIMENSION_X, 32>(cntP);
    cntN = blockSum<BLOCK_DIMENSION_X, 32>(cntN);

    if (threadIdx.x == 0) {
        atomicAdd(sum + 0, cntP);
        atomicAdd(sum + 1, cntN);
    }
}


// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
static void countSpins(const int redBlocks,
                       const size_t total_words,
                       const unsigned long long *d_black_tiles,
                       unsigned long long *d_sum,
                       unsigned long long *black_sum,
                       unsigned long long *white_sum) {
    CHECK_CUDA(cudaMemset(d_sum, 0, 2 * sizeof(*d_sum)))
    // Only the pointer to the black tiles is needed, since it provides access
    // to all spins (d_spins).
    // see definition in kernel.cu:
    // 		d_black_tiles = d_spins;
    // 		d_white_tiles = d_spins + total_words / 2;
    computeMagnetisation<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(
            total_words, d_black_tiles, d_sum
    );
    CHECK_ERROR("computeMagnetisation")
    CHECK_CUDA(cudaDeviceSynchronize())

    black_sum[0] = 0;
    white_sum[0] = 0;

    unsigned long long sum_h[2];

    CHECK_CUDA(cudaMemcpy(sum_h, d_sum, 2 * sizeof(*sum_h), cudaMemcpyDeviceToHost))
    black_sum[0] += sum_h[0];
    white_sum[0] += sum_h[1];
}

/**
 * @brief Read an existing lattice state from a file.
 *
 * @param d_spins  A pointer to the device array of spins to read in from the file
 * @param filename  The name of the file containing the lattice state
 * @param nrows  The number of rows in the lattice configuration
 * @param ncolumns  The number of columns in the lattice configuration
 * @param total_number_of_words  The total number of computer words storing the lattice configuration
 *
 */
void readFromFile(unsigned long long* d_spins,
                  const char* filename,
                  const size_t nrows,
                  const size_t ncolumns,
                  const size_t total_number_of_words) {
    auto ONE = static_cast<unsigned long long>(1);

    unsigned long long *h_spins = (unsigned long long *) malloc(total_number_of_words * sizeof(*h_spins));
    memset(h_spins, 0, total_number_of_words * sizeof(*h_spins));

    unsigned long long *h_black_tiles = h_spins;
    unsigned long long *h_white_tiles = h_spins + total_number_of_words / 2;

    FILE *fp;
    char buff[255];

    fp = fopen(filename, "r");

    for(int i = 0; i < nrows; i++) {
        for(int j = 0; j < ncolumns; j++) {
            for(int k = 0; k < 8 * sizeof(*h_spins); k += BIT_X_SPIN) {
                if (i & 1) {
                    fscanf(fp, "%s", buff);
                    if (std::atoi(buff))
                        h_white_tiles[i * ncolumns + j] |= ONE << k;
                    fscanf(fp, "%s", buff);
                    if (std::atoi(buff))
                        h_black_tiles[i * ncolumns + j] |= ONE << k;
                } else {
                    fscanf(fp, "%s", buff);
                    if (std::atoi(buff))
                        h_black_tiles[i * ncolumns + j] |= ONE << k;
                    fscanf(fp, "%s", buff);
                    if (std::atoi(buff))
                        h_white_tiles[i * ncolumns + j] |= ONE << k;
                }
            }
        }
    }
    fclose(fp);
    CHECK_CUDA(cudaMemcpy(d_spins, h_spins, total_number_of_words * sizeof(*h_spins),
                          cudaMemcpyHostToDevice))
    free(h_spins);
}


/**
 * @brief Read an existing lattice state from a binary file.
 *
 * Read in the spins stored in 32 bit words with little endian.
 *
 * @param d_spins  A pointer to the device array of spins to read in from the file
 * @param filename  The name of the file containing the lattice state
 * @param nrows  The number of rows in the lattice configuration
 * @param ncolumns  The number of columns in the lattice configuration
 * @param total_number_of_words  The total number of computer words storing the lattice configuration
 *
 */
void readFromFileBinary(unsigned long long* d_spins,
                  const char* filename,
                  const size_t nrows,
                  const size_t ncolumns,
                  const size_t total_number_of_words) {
    auto ONE = static_cast<unsigned long long>(1);

    unsigned long long *h_spins = (unsigned long long *) malloc(total_number_of_words * sizeof(*h_spins));
    memset(h_spins, 0, total_number_of_words * sizeof(*h_spins));

    unsigned long long *h_black_tiles = h_spins;
    unsigned long long *h_white_tiles = h_spins + total_number_of_words / 2;

    FILE *fp;
    fp = fopen(filename, "rb");

    // the buffer needs to be exactly of 2 * 8 * sizeof(*h_spins) / BIT_X_SPIN
    // for the read in to be correct
    int32_t buffer = 0;
    int bits_read_from_buffer = 0;
    for(int i = 0; i < nrows; i++) {
        for(int j = 0; j < ncolumns; j++) {
            fread(&buffer,sizeof(buffer),1,fp);
            bits_read_from_buffer = 0;
            // the words are stored in little endian meaning
            // the first spin we want to retrieve is located
            // on the right
            for(int k = 0; k < 8 * sizeof(*h_spins); k += BIT_X_SPIN) {
                if (i & 1) {
                    if ((buffer >> (31 - bits_read_from_buffer)) & 1)
                        h_white_tiles[i * ncolumns + j] |= ONE << k;
                    if ((buffer >> (31 - bits_read_from_buffer - 1)) & 1)
                        h_black_tiles[i * ncolumns + j] |= ONE << k;
                } else {
                    if ((buffer >> (31 - bits_read_from_buffer)) & 1)
                        h_black_tiles[i * ncolumns + j] |= ONE << k;
                    if ((buffer >> (31 - bits_read_from_buffer - 1)) & 1)
                        h_white_tiles[i * ncolumns + j] |= ONE << k;
                }
                bits_read_from_buffer += 2;
            }
        }
    }
    fclose(fp);
    CHECK_CUDA(cudaMemcpy(d_spins, h_spins, total_number_of_words * sizeof(*h_spins),
                          cudaMemcpyHostToDevice))
    free(h_spins);
}


/**
 * @brief Save the current lattice state to a binary file
 *
 * Caution: When reading in the file remember that the spins are stored
 *          in a 4 byte word saved in little endian!
 *
 * @param filename  The name of the file to save the lattice to
 * @param nrows  The number of rows in the lattice
 * @param ncolumns  The number of columns in the lattice
 * @param total_number_of_words  The total number of computer words storing the lattice configuration
 * @param d_spins  The pointer to the array storing the spins
 *
 */
static void dumpLatticeBinary(const char* filename,
                        const int nrows,
                        const size_t ncolumns,
                        const size_t total_number_of_words,
                        const unsigned long long *d_spins) {
    unsigned long long *v_h = (unsigned long long *) malloc(total_number_of_words * sizeof(*v_h));
    CHECK_CUDA(cudaMemcpy(v_h, d_spins, total_number_of_words * sizeof(*v_h),
                          cudaMemcpyDeviceToHost))

    unsigned long long *black_h = v_h;
    unsigned long long *white_h = v_h + total_number_of_words / 2;

    FILE *fp = fopen(filename, "wb");
    // the buffer needs to be exactly of 2 * 8 * sizeof(*h_spins) / BIT_X_SPIN
    // for the output to be correct
    int32_t buffer = 0;
    int NBIT = 32;
    int times_written_to_buffer = 0;

    for(int i = 0; i < nrows; i++) {
        for(int j = 0; j < ncolumns; j++) {
            unsigned long long b = black_h[i * ncolumns + j];
            unsigned long long w = white_h[i * ncolumns + j];

            for(int k = 0; k < 8 * sizeof(*v_h); k += BIT_X_SPIN) {
                int32_t black_spin = (b >> k) & 0xF;
                int32_t white_spin = (w >> k) & 0xF;
                if (i & 1) {
                    buffer |= white_spin << (NBIT - times_written_to_buffer - 1);
                    buffer |= black_spin << (NBIT - times_written_to_buffer - 2);
                } else {
                    buffer |= black_spin << (NBIT - times_written_to_buffer - 1);
                    buffer |= white_spin << (NBIT - times_written_to_buffer - 2);
                }
                times_written_to_buffer += 2;
            }
            fwrite(&buffer,sizeof(buffer),1,fp);
            buffer = 0;
            times_written_to_buffer = 0;
        }
    }
    fclose(fp);
    free(v_h);
}


/**
 * @brief Save the current lattice state to a file
 *
 * @param filename  The name of the file to save the lattice to
 * @param nrows  The number of rows in the lattice
 * @param ncolumns  The number of columns in the lattice
 * @param total_number_of_words  The total number of computer words storing the lattice configuration
 * @param d_spins  The pointer to the array storing the spins
 *
 */
static void dumpLattice(const char* filename,
                        const int nrows,
                        const size_t ncolumns,
                        const size_t total_number_of_words,
                        const unsigned long long *d_spins) {
    unsigned long long *v_h = (unsigned long long *) malloc(total_number_of_words * sizeof(*v_h));
    CHECK_CUDA(cudaMemcpy(v_h, d_spins, total_number_of_words * sizeof(*v_h),
                          cudaMemcpyDeviceToHost))

    unsigned long long *black_h = v_h;
    unsigned long long *white_h = v_h + total_number_of_words / 2;

    FILE *fp = fopen(filename, "w");

    for(int i = 0; i < nrows; i++) {
        for(int j = 0; j < ncolumns; j++) {
            unsigned long long b = black_h[i * ncolumns + j];
            unsigned long long w = white_h[i * ncolumns + j];

            for(int k = 0; k < 8 * sizeof(*v_h); k += BIT_X_SPIN) {
                if (i & 1) {
                    fprintf(fp, "%llX ", (w >> k) & 0xF);
                    fprintf(fp, "%llX ", (b >> k) & 0xF);
                } else {
                    fprintf(fp, "%llX ", (b >> k) & 0xF);
                    fprintf(fp, "%llX ", (w >> k) & 0xF);
                }
            }
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    free(v_h);
}


/**
 * @brief Perform one full lattice update
 *
 * Update the whole lattice by subsequently updating the "black" and "white" tiles of the lattice.
 *
 * @param iteration  The time of the current iteration
 * @param blocks  The number of blocks to be launched on the GPU
 * @param threads_per_block  The threads per block to be launched on the GPU
 * @param reduce_blocks  The number of blocks to be used during the summation
 * @param d_black_tiles  The pointer to the "black" tiles
 * @param d_white_tiles  The pointer to the "white" tiles
 * @param d_sum  The pointer to the array storing the sum over all spins
 * @param d_probabilities  The pointer to the array that will be storing the probabilities
 * @param params  A struct storing various simulation parameters.
 *
 * @return The relative magnetisation of the system
 *
 */
float update(int iteration,
             dim3 blocks, dim3 threads_per_block, const int reduce_blocks,
             unsigned long long *d_black_tiles,
             unsigned long long *d_white_tiles,
             unsigned long long *d_sum,
             float *d_probabilities,
             Parameters params) {
    unsigned long long spins_up;
    unsigned long long spins_down;
    countSpins(reduce_blocks, params.total_words, d_black_tiles, d_sum, &spins_up, &spins_down);
    double magnetisation = static_cast<double>(spins_up) - static_cast<double>(spins_down);
    float reduced_magnetisation = magnetisation / static_cast<double>(params.lattice_width * params.lattice_height);
    float market_coupling = params.reduced_alpha * fabs(reduced_magnetisation);
    precomputeProbabilities(d_probabilities, market_coupling, params.reduced_j, params.pitch);

    updateStrategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_BLACK, unsigned long long>
    <<<blocks, threads_per_block>>>
            (params.seed, iteration + 1,
             params.words_per_row / 2,
             params.lattice_height,
             reinterpret_cast<float (*)[5]>(d_probabilities),
             reinterpret_cast<ulonglong2 *>(d_white_tiles),
             reinterpret_cast<ulonglong2 *>(d_black_tiles));

    updateStrategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_WHITE, unsigned long long>
    <<<blocks, threads_per_block>>>
            (params.seed, iteration + 1,
             params.words_per_row / 2,
             params.lattice_height,
             reinterpret_cast<float (*)[5]>(d_probabilities),
             reinterpret_cast<ulonglong2 *>(d_black_tiles),
             reinterpret_cast<ulonglong2 *>(d_white_tiles));

    return reduced_magnetisation;
}

#endif