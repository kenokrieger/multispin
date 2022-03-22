#ifndef traders_cuh
#define traders_cuh

#include <curand_kernel.h>
#include "cudamacro.h"

#define BIT_X_SPIN (4)
#define THREADS (128)

#define BLOCK_DIMENSION_X_DEFINE (16)
#define BLOCK_DIMENSION_Y_DEFINE (16)

enum {C_BLACK, C_WHITE};

typedef struct {
    unsigned long long seed;
    float reduced_alpha;
    float reduced_j;
    long long lattice_height;
    long long lattice_width;
    size_t words_per_row;
    size_t total_words;
    size_t pitch;
} Parameters;


// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
__device__ __forceinline__ unsigned long long int custom_popc(const unsigned long long int x) {return __popcll(x);}

// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
__device__ __forceinline__ ulonglong2 custom_make_int2(const unsigned long long x, const unsigned long long y) {return make_ulonglong2(x, y);}


template<int BITXSPIN, int COLOR, typename INT_T, typename INT2_T>
__global__  void initialise_traders(const unsigned long long seed,
                                    const long long ncolumns,
                                    INT2_T *__restrict__ traders,
                                    float percentage = 0.5f) {
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto index = row * ncolumns + col;
    const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSPIN;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, index, static_cast<long long>(2 * SPIN_X_WORD) * COLOR, &rng);

    traders[index] = custom_make_int2(INT_T(0), INT_T(0));
    for(int spin_position = 0; spin_position < 8 * sizeof(INT_T); spin_position += BITXSPIN) {
        // The two if clauses are not identical since curand_uniform()
        // returns a different number on each invocation
        if (curand_uniform(&rng) < percentage) {
            traders[index].x |= INT_T(1) << spin_position;
        }
        if (curand_uniform(&rng) < percentage) {
            traders[index].y |= INT_T(1) << spin_position;
        }
    }
}


template<typename INT_T, typename INT2_T>
void initialise_arrays(dim3 blocks, dim3 threads_per_block,
                       const unsigned long long seed, const unsigned long long ncolumns,
                       INT2_T *__restrict__ d_black_tiles, INT2_T *__restrict__ d_white_tiles,
                       float percentage = 0.5f) {
    initialise_traders<BIT_X_SPIN, C_BLACK, INT_T><<<blocks, threads_per_block>>>(
            seed, ncolumns, reinterpret_cast<ulonglong2 *>(d_black_tiles), percentage
    );
    CHECK_ERROR("initialise_traders")

    initialise_traders<BIT_X_SPIN, C_WHITE, INT_T><<<blocks, threads_per_block>>>(
            seed, ncolumns, reinterpret_cast<ulonglong2 *>(d_white_tiles), percentage
    );
    CHECK_ERROR("initialise_traders")
}


template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y, typename INT2_T>
__device__ void load_tiles(const int ncolumns, const int nrows,
                           const INT2_T *__restrict__ traders, INT2_T tile[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2]) {
    const unsigned int tidx = threadIdx.x;
    const unsigned int tidy = threadIdx.y;

    const int tile_start_x = blockIdx.x * BLOCK_SIZE_X;
    const int tile_start_y = blockIdx.y * BLOCK_SIZE_Y;

    int row = tile_start_y + tidy;
    int col = tile_start_x + tidx;
    tile[1 + tidy][1 + tidx] = traders[row * ncolumns + col];

    if (tidy == 0) {
        row = (tile_start_y % nrows) == 0 ? nrows - 1 : tile_start_y - 1;
        tile[0][1 + tidx] = traders[row * ncolumns + col];

        row = (tile_start_y + BLOCK_SIZE_Y) % nrows;
        tile[1 + BLOCK_SIZE_Y][1 + tidx] = traders[row * ncolumns + col];

        row = tile_start_y + tidx;
        col = (tile_start_x % ncolumns) == 0 ? ncolumns - 1 : tile_start_x - 1;
        tile[1 + tidx][0] = traders[row * ncolumns + col];

        row = tile_start_y + tidx;
        col = (tile_start_x + BLOCK_SIZE_X) % ncolumns;
        tile[1 + tidx][1 + BLOCK_SIZE_X] = traders[row * ncolumns + col];
    }
}


__device__ void load_probabilities(const float precomputed_probabilities[][5], float shared_probabilities[][5]) {
    const unsigned tidx = threadIdx.x;
    const unsigned tidy = threadIdx.y;
    // load precomputed probabilities into shared memory.
    #pragma unroll
    for(unsigned i = 0; i < 2; i += blockDim.y) {
        #pragma unroll
        for(unsigned j = 0; j < 5; j += blockDim.x) {
            if (i + tidy < 2 && j + tidx < 5)
                shared_probabilities[i + tidy][j + tidx] = precomputed_probabilities[i + tidy][j + tidx];
        }
    }
}


template<int BLOCK_DIMENSION_X, int BITXSPIN, typename INT_T, typename INT2_T>
__device__ INT2_T compute_neighbour_sum(INT2_T shared_tiles[][BLOCK_DIMENSION_X + 2],
                                        const int tidx, const int tidy, const int shift_left) {
    // three nearest neighbors
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


template<int BITXSPIN, typename INT_T, typename INT2_T>
__device__ INT2_T flip_spins(curandStatePhilox4_32_10_t rng, INT2_T target, INT2_T parallel_sum, const float shared_probabilities[][5]) {
    const auto ONE = static_cast<INT_T>(1);
    #pragma unroll
    for(int spin_position = 0; spin_position < 8 * sizeof(INT_T); spin_position += BITXSPIN) {
        const int2 spin = make_int2((target.x >> spin_position) & 0xF, (target.y >> spin_position) & 0xF);
        const int2 sum = make_int2((parallel_sum.x >> spin_position) & 0xF, (parallel_sum.y >> spin_position) & 0xF);

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


template<int BLOCK_DIMENSION_X, int BLOCK_DIMENSION_Y, int BITXSPIN, int COLOR, typename INT_T, typename INT2_T>
__global__ void update_strategies(const unsigned long long seed, const int rng_invocations,
                                  const int ncolumns,
                                  const int nrows,
                                  const float precomputed_probabilities[][5],
                                  const INT2_T *__restrict__ checkerboard_agents,
                                  INT2_T *__restrict__ traders)
{
    const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSPIN;
    const unsigned tidx = threadIdx.x;
    const unsigned tidy = threadIdx.y;
    __shared__ INT2_T shared_tiles[BLOCK_DIMENSION_Y + 2][BLOCK_DIMENSION_X + 2];
    load_tiles<BLOCK_DIMENSION_X, BLOCK_DIMENSION_Y, INT2_T>(ncolumns, nrows,
                                                             checkerboard_agents, shared_tiles);
    __shared__ float shared_probabilities[2][5];
    load_probabilities(precomputed_probabilities, shared_probabilities);
    __syncthreads();

    int row = blockIdx.y * BLOCK_DIMENSION_Y + tidy;
    int col = blockIdx.x * BLOCK_DIMENSION_X + tidx;
    const int shift_left = (COLOR == C_BLACK) == !(row % 2);
    const long long index = row * ncolumns + col;
    // compute neighbor sum
    INT2_T parallel_sum = compute_neighbour_sum<BLOCK_DIMENSION_X, BITXSPIN, INT_T>(shared_tiles, tidx, tidy,
                                                                                    shift_left);
    // flip spin according to neighbor sum and its own orientation
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, index, static_cast<long long>(2 * SPIN_X_WORD) * (2 * rng_invocations + COLOR), &rng);
    traders[index] = flip_spins<BITXSPIN, INT_T>(rng, traders[index], parallel_sum, shared_probabilities);
}


void precompute_probabilities(float* probabilities, const float market_coupling, const float reduced_j, const size_t pitch)
{

    float h_probabilities[2][5];

    for (int spin = 0; spin < 2; spin++) {
        for (int idx = 0; idx < 5; idx++) {
            int neighbour_sum = 2 * idx - 4;
            float field = reduced_j * neighbour_sum + market_coupling * ((spin) ? 1 : -1);
            h_probabilities[spin][idx] = 1.0 / (1.0 + exp(field));
        }
    }
    CHECK_CUDA(cudaMemcpy2D(probabilities, 5 * sizeof(*h_probabilities), &h_probabilities, pitch, 5 * sizeof(*h_probabilities), 2, cudaMemcpyHostToDevice))
}


// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
template<int BLOCK_DIMENSION_X, int WSIZE, typename T>
__device__ __forceinline__ T block_sum(T traders)
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
__global__ void getMagn_k(const long long n,
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
    cntP = block_sum<BLOCK_DIMENSION_X, 32>(cntP);
    cntN = block_sum<BLOCK_DIMENSION_X, 32>(cntN);

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
                       unsigned long long *bsum,
                       unsigned long long *wsum)
{
    CHECK_CUDA(cudaMemset(d_sum, 0, 2 * sizeof(*d_sum)))
    // Only the pointer to the black tiles is needed, since it provides access
    // to all spins (d_spins).
    // see definition in kernel.cu:
    // 		d_black_tiles = d_spins;
    // 		d_white_tiles = d_spins + total_words / 2;
    getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(total_words, d_black_tiles, d_sum);
    CHECK_ERROR("getMagn_k")
    CHECK_CUDA(cudaDeviceSynchronize())

    bsum[0] = 0;
    wsum[0] = 0;

    unsigned long long sum_h[2];

    CHECK_CUDA(cudaMemcpy(sum_h, d_sum, 2 * sizeof(*sum_h), cudaMemcpyDeviceToHost))
    bsum[0] += sum_h[0];
    wsum[0] += sum_h[1];
}


__attribute__((unused)) static void dumpLattice(const long long iteration,
                        const int rows,
                        const size_t columns,
                        const size_t total_number_of_words,
                        const unsigned long long *v_d) {

    char filename[256];

    unsigned long long *v_h = (unsigned long long *) malloc(total_number_of_words * sizeof(*v_h));
    CHECK_CUDA(cudaMemcpy(v_h, v_d, total_number_of_words * sizeof(*v_h), cudaMemcpyDeviceToHost))

    unsigned long long *black_h = v_h;
    unsigned long long *white_h = v_h + total_number_of_words / 2;

    snprintf(filename, sizeof(filename), "iteration_%lld.dat", iteration);
    FILE *fp = fopen(filename, "w");

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            unsigned long long b = black_h[i * columns + j];
            unsigned long long w = white_h[i * columns + j];

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


template<int SPIN_X_WORD>
float update(int iteration,
             dim3 blocks, dim3 threads_per_block, const int reduce_blocks,
             unsigned long long *d_black_tiles,
             unsigned long long *d_white_tiles,
             unsigned long long *d_sum,
             float *d_probabilities,
             unsigned long long spins_up,
             unsigned long long spins_down,
             Parameters params)
{
    countSpins(reduce_blocks, params.total_words, d_black_tiles, d_sum, &spins_up, &spins_down);
    double magnetisation = static_cast<double>(spins_up) - static_cast<double>(spins_down);
    float reduced_magnetisation = magnetisation / static_cast<double>(params.lattice_width * params.lattice_height);
    float market_coupling = -params.reduced_alpha * fabs(reduced_magnetisation);
    precompute_probabilities(d_probabilities, market_coupling, params.reduced_j, params.pitch);

    update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_BLACK, unsigned long long>
    <<<blocks, threads_per_block>>>
            (params.seed, iteration + 1,
             params.words_per_row / 2,
             params.lattice_height,
             reinterpret_cast<float (*)[5]>(d_probabilities),
             reinterpret_cast<ulonglong2 *>(d_white_tiles),
             reinterpret_cast<ulonglong2 *>(d_black_tiles));

    update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_WHITE, unsigned long long>
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