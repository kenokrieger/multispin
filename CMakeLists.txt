cmake_minimum_required(VERSION 3.21)
project(multising CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(multising source/kernel.cu)

set_target_properties(multising PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "61;75")
