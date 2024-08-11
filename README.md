# par-trans: Parallel transport on Matrix Manifolds

This project provides examples of Parallel transport on matrix manifolds. 
* We provide an $O(nd^2)$ algorithm for short transport time, and $O(tp^3)$ for long transport time for the Stiefel manifold $St_{n, p}$ with a family of metrics defined by a parameter $\alpha$ as in [1], (defined previously in [2] with a different parametrization). This contributes to a problem raised in [1].
* For flag manifolds with the canonical metric, our algorithm has the same complexity.
* We provide closed-form parallel transport using exponential action for GL(n) and SO(n), with related families of metrics.

# Installation:
To install from git do (assume you have *build*, otherwise do pip install build).

```
!pip install git+https://github.com/dnguyend/par-trans
```
Alternatively, you can clone the project to your local directory and then add the directory to your PYTHONPATH. If you only want to run the numpy version, this may be an option.
Check the [documentation page](https://dnguyend.github.io/par-trans/index.html). Look at the [examples](https://github.com/dnguyend/par-trans/tree/main/examples) and the [tests](https://github.com/dnguyend/par-trans/tree/main/tests).

References\
[1] A. Edelman, T. A. Arias, and S. T. Smith, The geometry of algorithms with orthogonality constraints, SIAM J. Matrix Anal. Appl., 20 (1999), pp. 303–353\
[2] K. H&#252;per, I. Markina, and F. Silva Leite, A Lagrangian approach to extremal curves on Stiefel manifolds, Journal of Geometric Mechanics, 13 (2021), pp. 55–72, https://doi.org/10.3934/jgm.2020031.788\

[3] D. Nguyen, Operator-valued formulas for Riemannian gradient and Hessian and families of tractable metrics in Riemannian optimization, Journal of Optimization Theory and Applications, 198 (2023), pp. 135–164, https://doi.org/https://doi.org/10.1007/815
s10957-023-02242-z.816
