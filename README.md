# Multising

Multising is the GPU implementation of the Bornholdt Ising Model in CUDA C. All
background information can be found at the
[main page](https://github.com/kenokrieger/multising).

## Compiling

This is project was created using Jetbrain's CLion IDE. The easiest way to 
compile it, is by loading it into CLion. Since it only consists of three
files, you may also try to compile it manually using the nvcc compiler or
CMake.

## Usage

The program expects a file "multising.conf" in the directory it is called from.
This file contains all the values for the simulation like grid size and parameters.
The path to the file can also be passed as the first argument in the terminal.

Example configuration file:

```
lattice_height = 2048
lattice_width = 2048
total_updates = 100000
seed = 2458242462
alpha = 15.0
j = 1.0
beta = 0.6
init_up = 0.5
rng_offset = 0
import = initial_state.dat
export = final_state.dat
```

For **alpha = 0** this model resembles the standard Ising model. The fields
**rng_offset** and **import** can be used to continue a simulation.

## License

This project is licensed under MIT License (see LICENSE.txt).
