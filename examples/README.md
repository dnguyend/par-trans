# Examples and location of numerical experiment results
This directory contains workbooks showing the syntax for calling the parallel transport functions for the manifolds discussed in the paper. Each manifold is implemented as a class, as showed the [documentation page](https://dnguyend.github.io/par-trans/index.html). 
There are two submodules, one implemented in JAX, containing Stiefel and Flag manifolds. The other contains the Numpy implementation for all four manifolds. The workbook names contain the submodule names. In the JAX version, we can 
use Automatic Differentiation to verify that the transport equations are solved at high accuracy. It also works with GPU if connected to a GPU session (the free colab GPU session has a limit in accessing time).
## Tables and Graphs for the paper
The computations are in the Numpy workbooks, toward the end.
## Hardware
The hardware configuration can be viewed at the end of the workbook [NumpyFlagParallel.ipynb](https://github.com/dnguyend/par-trans/blob/main/examples/NumpyFlagParallel.ipynb).
## Accuracy
We show one example near the beginning of the workbook. Later on, the columns labeled as *log_err_eq* in the tables at the end of [JAXStiefelParallel.ipynb](https://github.com/dnguyend/par-trans/blob/main/examples/JAXStiefelParallel.ipynb) and [JAXFlagParallel.ipynb](https://github.com/dnguyend/par-trans/blob/main/examples/JAXFlagParallel.ipynb)
show the accuracy of the solution. As mentioned, the JAX version allows a more accurate evaluation of the derivative. Numerical derivatives using finite difference are also shown in the numpy version.
## Time improvement of the custom calculation of norm.
This block of results at the end of [NumpyStiefelParallel.ipynb](https://github.com/dnguyend/par-trans/blob/main/examples/NumpyStiefelParallel.ipynb) shows the time improvement.

Time scipy expm_multiply 5.082546\
Time expm_multiply use only 1-norm 5.523014\
Time using solve_w 0.458826\
Time ratio between expm_multipy and solve_w 11.077296

Earlier "time expm_multiply" versus "time solv_w" near the beginning of the workbook shows the total time comparison (including the aggregation with matrix exponential).

A similar result is shown for flag manifolds in [NumpyFlagParallel.ipynb](https://github.com/dnguyend/par-trans/blob/main/examples/NumpyFlagParallel.ipynb).
